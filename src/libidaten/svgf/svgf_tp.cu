#include "svgf/svgf_pt.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/compaction.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

//#define ENABLE_MEDIAN_FILTER

inline __device__ void computePrevScreenPos(
	int ix, int iy,
	float centerDepth,
	int width, int height,
	aten::vec4* prevPos,
	const aten::mat4* __restrict__ mtxs)
{
	// NOTE
	// Pview = (Xview, Yview, Zview, 1)
	// mtxV2C = W 0 0  0
	//          0 H 0  0
	//          0 0 A  B
	//          0 0 -1 0
	// mtxV2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, Zview)
	//  Wclip = Zview = depth
	// Xscr = Xclip / Wclip = Xclip / Zview = Xclip / depth
	// Yscr = Yclip / Wclip = Yclip / Zview = Yclip / depth
	//
	// Xscr * depth = Xclip
	// Xview = mtxC2V * Xclip

	const aten::mat4 mtxC2V = mtxs[0];
	const aten::mat4 mtxPrevV2C = mtxs[1];

	float2 uv = make_float2(ix + 0.5, iy + 0.5);
	uv /= make_float2(width - 1, height - 1);	// [0, 1]
	uv = uv * 2.0f - 1.0f;	// [0, 1] -> [-1, 1]

	aten::vec4 pos(uv.x, uv.y, 0, 0);

	// Screen-space -> Clip-space.
	pos.x *= centerDepth;
	pos.y *= centerDepth;

	// Clip-space -> View-space
	pos = mtxC2V.apply(pos);
	pos.z = -centerDepth;
	pos.w = 1.0;

	// Reproject previous screen position.
	*prevPos = mtxPrevV2C.apply(pos);
	*prevPos /= prevPos->w;

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

__global__ void temporalReprojection(
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	const aten::CameraParameter* __restrict__ camera,
	idaten::SVGFPathTracing::AOV* curAovs,
	idaten::SVGFPathTracing::AOV* prevAovs,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t dst,
	float4* tmpBuffer,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	const float centerDepth = curAovs[idx].depth;
	const int centerMeshId = curAovs[idx].meshid;

	// 今回のフレームのピクセルカラー.
	float4 curColor = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	curColor.w = 1;

	if (centerMeshId < 0) {
		// 背景なので、そのまま出力して終わり.
		surf2Dwrite(
			curColor,
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		curAovs[idx].color = curColor;
		curAovs[idx].moments = make_float4(1);

		return;
	}

	auto centerNormal = curAovs[idx].normal;

	float4 sum = make_float4(0, 0, 0, 0);
	float weight = 0.0f;

	static const float zThreshold = 0.05f;
	static const float nThreshold = 0.98f;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			int _idx = getIdx(xx, yy, width);

			auto depth = curAovs[_idx].depth;

			// 前のフレームのクリップ空間座標を計算.
			aten::vec4 prevPos;
			computePrevScreenPos(
				xx, yy,
				depth,
				width, height,
				&prevPos,
				mtxs);

			// [0, 1]の範囲内に入っているか.
			bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
			bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

			if (isInsideX && isInsideY) {
				// 前のフレームのスクリーン座標.
				int px = (int)(prevPos.x * width - 0.5f);
				int py = (int)(prevPos.y * height - 0.5f);

				px = clamp(px, 0, width - 1);
				py = clamp(py, 0, height - 1);

				int pidx = getIdx(px, py, width);

				const float prevDepth = prevAovs[pidx].depth;
				const int prevMeshId = prevAovs[pidx].meshid;

				auto prevNormal = prevAovs[pidx].normal;

				// TODO
				// 同じメッシュ上でもライトのそばの明るくなったピクセルを拾ってしまう場合の対策が必要.

				float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
				float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
				float Wm = centerMeshId == prevMeshId ? 1.0f : 0.0f;

				// 前のフレームのピクセルカラーを取得.
				float4 prev = prevAovs[pidx].color;

				float W = Wz * Wn * Wm;
				sum += prev * W;
				weight += W;
			}
		}
	}
	
	if (weight > 0.0f) {
		sum /= weight;
		weight /= 9;
#if 0
		curColor = 0.2 * curColor + 0.8 * sum;
#else
		curColor = (1.0f - weight) * curColor + weight * sum;
#endif
	}

	curAovs[idx].temporalWeight = weight;

#ifdef ENABLE_MEDIAN_FILTER
	tmpBuffer[idx] = curColor;
#else
	curAovs[idx].color = curColor;

	// TODO
	// 現フレームと過去フレームが同率で加算されるため、どちらかに強い影響がでると影響が弱まるまでに非常に時間がかかる.
	// ex) 
	// f0 = 100, f1 = 0, f2 = 0
	// avg = (f0 + f1 + f2) / 3 = 33.3 <- 非常に大きい値が残り続ける.

	// accumulate moments.
	{
		float lum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);
		float4 centerMoment = make_float4(lum * lum, lum, 0, 0);

		// 前のフレームのクリップ空間座標を計算.
		aten::vec4 prevPos;
		computePrevScreenPos(
			ix, iy,
			centerDepth,
			width, height,
			&prevPos,
			mtxs);

		// [0, 1]の範囲内に入っているか.
		bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
		bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

		// 積算フレーム数のリセット.
		int frame = 1;

		if (isInsideX && isInsideY) {
			int px = (int)(prevPos.x * width - 0.5f);
			int py = (int)(prevPos.y * height - 0.5f);

			px = clamp(px, 0, width - 1);
			py = clamp(py, 0, height - 1);

			int pidx = getIdx(px, py, width);

			const float prevDepth = prevAovs[pidx].depth;
			const int prevMeshId = prevAovs[pidx].meshid;

			auto prevNormal = prevAovs[pidx].normal;

			if (abs(1 - centerDepth / prevDepth) < zThreshold
				&& dot(centerNormal, prevNormal) > nThreshold
				&& centerMeshId == prevMeshId)
			{
				float4 prevMoment = prevAovs[pidx].moments;

				// 積算フレーム数を１増やす.
				frame = (int)prevMoment.w + 1;

				centerMoment += prevMoment;
			}
		}

		centerMoment.w = frame;

		curAovs[idx].moments = centerMoment;
	}
#endif

	surf2Dwrite(
		curColor,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

__global__ void dilateWeight(
	idaten::SVGFPathTracing::AOV* curAovs,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	auto idx = getIdx(ix, iy, width);

	const int centerMeshId = curAovs[idx].meshid;

	if (centerMeshId < 0) {
		// This pixel is background, so nothing is done.
		return;
	}

	float temporalWeight = curAovs[idx].temporalWeight;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = ix + x;
			int yy = iy + y;

			if ((0 <= xx) && (xx < width)
				&& (0 <= yy) && (yy < height))
			{
				int pidx = getIdx(xx, yy, width);
				float w = curAovs[pidx].temporalWeight;
				temporalWeight = min(temporalWeight, w);
			}
		}
	}

	curAovs[idx].temporalWeight = temporalWeight;
}

inline __device__ float4 min(float4 a, float4 b)
{
	return make_float4(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z),
		min(a.w, b.w));
}

inline __device__ float4 max(float4 a, float4 b)
{
	return make_float4(
		max(a.x, b.x),
		max(a.y, b.y),
		max(a.z, b.z),
		max(a.w, b.w));
}

// Macro for sorting.
#define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)			s2(a, b); s2(a, c);
#define mx3(a, b, c)			s2(b, c); s2(a, c);

#define mnmx3(a, b, c)			mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)		s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)	s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

template <bool isReferPath>
inline __device__ float4 medianFilter(
	int ix, int iy,
	const float4* src,
	const idaten::SVGFPathTracing::Path* paths,
	int width, int height)
{
	float4 v[9];

	int pos = 0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			int pidx = getIdx(xx, yy, width);

			if (isReferPath) {
				v[pos] = make_float4(paths[pidx].contrib.x, paths[pidx].contrib.y, paths[pidx].contrib.z, 0);
			}
			else {
				v[pos] = src[pidx];
			}
			pos++;
		}
	}

	// Sort
	float4 temp;
	mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
	mnmx5(v[1], v[2], v[3], v[4], v[6]);
	mnmx4(v[2], v[3], v[4], v[7]);
	mnmx3(v[3], v[4], v[8]);

	return v[4];
}

__global__ void medianFilter(
	cudaSurfaceObject_t dst,
	const float4* __restrict__ src,
	idaten::SVGFPathTracing::AOV* curAovs,
	const idaten::SVGFPathTracing::AOV* __restrict__ prevAovs,
	const aten::mat4* __restrict__ mtxs,
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	auto idx = getIdx(ix, iy, width);

	const int centerMeshId = curAovs[idx].meshid;

	if (centerMeshId < 0) {
		// This pixel is background, so nothing is done.
		return;
	}

	auto curColor = medianFilter<false>(ix, iy, src, paths, width, height);

	curAovs[idx].color = curColor;

	const float centerDepth = curAovs[idx].depth;
	const auto centerNormal = curAovs[idx].normal;

	static const float zThreshold = 0.05f;
	static const float nThreshold = 0.98f;

	// accumulate moments.
	{
		float lum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);
		float4 centerMoment = make_float4(lum * lum, lum, 0, 0);

		// 前のフレームのクリップ空間座標を計算.
		aten::vec4 prevPos;
		computePrevScreenPos(
			ix, iy,
			centerDepth,
			width, height,
			&prevPos,
			mtxs);

		// [0, 1]の範囲内に入っているか.
		bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
		bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

		// 積算フレーム数のリセット.
		int frame = 1;

		if (isInsideX && isInsideY) {
			int px = (int)(prevPos.x * width - 0.5f);
			int py = (int)(prevPos.y * height - 0.5f);

			px = clamp(px, 0, width - 1);
			py = clamp(py, 0, height - 1);

			int pidx = getIdx(px, py, width);

			const float prevDepth = prevAovs[pidx].depth;
			const int prevMeshId = prevAovs[pidx].meshid;

			auto prevNormal = prevAovs[pidx].normal;

			if (abs(1 - centerDepth / prevDepth) < zThreshold
				&& dot(centerNormal, prevNormal) > nThreshold
				&& centerMeshId == prevMeshId)
			{
				float4 prevMoment = prevAovs[pidx].moments;

				// 積算フレーム数を１増やす.
				frame = (int)prevMoment.w + 1;

				centerMoment += prevMoment;
			}
		}

		centerMoment.w = frame;

		curAovs[idx].moments = centerMoment;
	}

	surf2Dwrite(
		curColor,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void SVGFPathTracing::onTemporalReprojection(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();
		auto& prevaov = getPrevAovs();

		aten::mat4 mtxs[2] = {
			m_mtxC2V,
			m_mtxPrevV2C,
		};

		m_mtxs.init(sizeof(aten::mat4) * AT_COUNTOF(mtxs));
		m_mtxs.writeByNum(mtxs, AT_COUNTOF(mtxs));

		temporalReprojection << <grid, block >> > (
		//temporalReprojection << <1, 1 >> > (
			m_paths.ptr(),
			m_cam.ptr(),
			curaov.ptr(),
			prevaov.ptr(),
			m_mtxs.ptr(),
			outputSurf,
			m_tmpBuf.ptr(),
			width, height);

		checkCudaKernel(temporalReprojection);

#ifdef ENABLE_MEDIAN_FILTER
		medianFilter << <grid, block >> > (
			outputSurf,
			m_tmpBuf.ptr(),
			curaov.ptr(),
			prevaov.ptr(),
			m_mtxs.ptr(),
			m_paths.ptr(),
			width, height);

		checkCudaKernel(medianFilter);
#endif

		dilateWeight << <grid, block >> > (
			curaov.ptr(),
			width, height);
		checkCudaKernel(dilateWeight);

		m_mtxs.reset();
	}
}
