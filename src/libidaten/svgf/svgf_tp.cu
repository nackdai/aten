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
	const aten::mat4 mtxV2W = mtxs[1];
	const aten::mat4 mtxPrevW2V = mtxs[2];
	const aten::mat4 mtxV2C = mtxs[3];

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

	pos = mtxV2W.apply(pos);

	// Reproject previous screen position.
	pos = mtxPrevW2V.apply(pos);
	*prevPos = mtxV2C.apply(pos);
	*prevPos /= prevPos->w;

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

inline __device__ int getLinearIdx(int x, int y, int w, int h)
{
	int max_buffer_size = w * h;
	return clamp(y * w + x, 0, max_buffer_size - 1);
}

// Bilinear sampler
inline __device__ float4 sampleBilinear(
	const float4* buffer, 
	float uvx, float uvy,
	int w, int h)
{
	float2 uv = make_float2(uvx, uvy) * make_float2(w, h) - make_float2(0.5f, 0.5f);

	int x = floor(uv.x);
	int y = floor(uv.y);

	float2 uv_ratio = uv - make_float2(x, y);
	float2 uv_inv = make_float2(1.f, 1.f) - uv_ratio;

	int x1 = clamp(x + 1, 0, w - 1);
	int y1 = clamp(y + 1, 0, h - 1);

	float4 r = (buffer[getLinearIdx(x, y, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y, w, h)] * uv_ratio.x) * uv_inv.y +
		(buffer[getLinearIdx(x, y1, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y1, w, h)] * uv_ratio.x) * uv_ratio.y;

	return r;
}

__global__ void temporalReprojection(
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	const aten::CameraParameter* __restrict__ camera,
	float4* curAovNormalDepth,
	float4* curAovTexclrTemporalWeight,
	float4* curAovColorVariance,
	float4* curAovMomentMeshid,
	const float4* __restrict__ prevAovNormalDepth,
	const float4* __restrict__ prevAovTexclrTemporalWeight,
	const float4* __restrict__ prevAovColorVariance,
	const float4* __restrict__ prevAovMomentMeshid,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t dst,
	float4* tmpBuffer,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	auto nmlDepth = curAovNormalDepth[idx];
	auto momentMeshId = curAovMomentMeshid[idx];

	const float centerDepth = nmlDepth.w;
	const int centerMeshId = (int)momentMeshId.w;

	// 今回のフレームのピクセルカラー.
	float4 curColor = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 1.0f) / path.samples;
	//curColor.w = 1;

	if (centerMeshId < 0) {
		// 背景なので、そのまま出力して終わり.
		surf2Dwrite(
			curColor,
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		curAovColorVariance[idx] = curColor;
		curAovMomentMeshid[idx] = make_float4(1, 1, 1, curAovMomentMeshid[idx].w);

		return;
	}

	float3 centerNormal = make_float3(nmlDepth.x, nmlDepth.y, nmlDepth.z);

	float4 sum = make_float4(0);
	float weight = 0.0f;

	static const float zThreshold = 0.05f;
	static const float nThreshold = 0.98f;

	aten::vec4 centerPrevPos;

#pragma unroll
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			int _idx = getIdx(xx, yy, width);

			nmlDepth = curAovNormalDepth[_idx];
			auto depth = nmlDepth.w;

			// 前のフレームのクリップ空間座標を計算.
			aten::vec4 prevPos;
			computePrevScreenPos(
				xx, yy,
				depth,
				width, height,
				&prevPos,
				mtxs);

			if (x == 0 && y == 0) {
				centerPrevPos = prevPos;
			}

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

				nmlDepth = prevAovNormalDepth[pidx];
				momentMeshId = prevAovMomentMeshid[pidx];

				const float prevDepth = nmlDepth.w;
				const int prevMeshId = (int)momentMeshId.w;
				float3 prevNormal = make_float3(nmlDepth.x, nmlDepth.y, nmlDepth.z);

				// TODO
				// 同じメッシュ上でもライトのそばの明るくなったピクセルを拾ってしまう場合の対策が必要.

				float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
				float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
				float Wm = centerMeshId == prevMeshId ? 1.0f : 0.0f;

				// 前のフレームのピクセルカラーを取得.
				//float4 prev = prevAovColorVariance[pidx];
				float4 prev = sampleBilinear(prevAovColorVariance, prevPos.x, prevPos.y, width, height);

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

	curAovTexclrTemporalWeight[idx].w = weight;

#ifdef ENABLE_MEDIAN_FILTER
	tmpBuffer[idx] = curColor;
#else
	curAovColorVariance[idx].x = curColor.x;
	curAovColorVariance[idx].y = curColor.y;
	curAovColorVariance[idx].z = curColor.z;

	// TODO
	// 現フレームと過去フレームが同率で加算されるため、どちらかに強い影響がでると影響が弱まるまでに非常に時間がかかる.
	// ex) 
	// f0 = 100, f1 = 0, f2 = 0
	// avg = (f0 + f1 + f2) / 3 = 33.3 <- 非常に大きい値が残り続ける.

	// accumulate moments.
	{
		float lum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);
		float3 centerMoment = make_float3(lum * lum, lum, 0);

#if 0
		// 前のフレームのクリップ空間座標を計算.
		aten::vec4 prevPos;
		computePrevScreenPos(
			ix, iy,
			centerDepth,
			width, height,
			&prevPos,
			mtxs);
#else
		aten::vec4 prevPos = centerPrevPos;
#endif

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

			nmlDepth = prevAovNormalDepth[pidx];
			momentMeshId = prevAovMomentMeshid[pidx];

			const float prevDepth = nmlDepth.w;
			const int prevMeshId = (int)momentMeshId.w;
			float3 prevNormal = make_float3(nmlDepth.x, nmlDepth.y, nmlDepth.z);

			if (abs(1 - centerDepth / prevDepth) < zThreshold
				&& dot(centerNormal, prevNormal) > nThreshold
				&& centerMeshId == prevMeshId)
			{
				auto momentMeshid = prevAovMomentMeshid[pidx];;
				float3 prevMoment = make_float3(momentMeshid.x, momentMeshid.y, momentMeshid.z);

				// 積算フレーム数を１増やす.
				frame = (int)prevMoment.z + 1;

				centerMoment += prevMoment;
			}
		}

		centerMoment.z = frame;

		curAovMomentMeshid[idx].x = centerMoment.x;
		curAovMomentMeshid[idx].y = centerMoment.y;
		curAovMomentMeshid[idx].z = centerMoment.z;
	}
#endif

	surf2Dwrite(
		curColor,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

__global__ void dilateWeight(
	float4* aovTexclrTemporalWeight,
	const float4* __restrict__ aovMomentMeshid,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	auto idx = getIdx(ix, iy, width);

	const int centerMeshId = (int)aovMomentMeshid[idx].w;

	if (centerMeshId < 0) {
		// This pixel is background, so nothing is done.
		return;
	}

	float temporalWeight = aovTexclrTemporalWeight[idx].w;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = ix + x;
			int yy = iy + y;

			if ((0 <= xx) && (xx < width)
				&& (0 <= yy) && (yy < height))
			{
				int pidx = getIdx(xx, yy, width);
				float w = aovTexclrTemporalWeight[pidx].w;
				temporalWeight = min(temporalWeight, w);
			}
		}
	}

	aovTexclrTemporalWeight[idx].w = temporalWeight;
}

#if 0
inline __device__ float3 min(float3 a, float3 b)
{
	return make_float3(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z));
}

inline __device__ float3 max(float3 a, float3 b)
{
	return make_float3(
		max(a.x, b.x),
		max(a.y, b.y),
		max(a.z, b.z));
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
inline __device__ float3 medianFilter(
	int ix, int iy,
	const float4* src,
	const idaten::SVGFPathTracing::Path* paths,
	int width, int height)
{
	float3 v[9];

	int pos = 0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			int pidx = getIdx(xx, yy, width);

			if (isReferPath) {
				v[pos] = make_float3(paths[pidx].contrib.x, paths[pidx].contrib.y, paths[pidx].contrib.z);
			}
			else {
				auto s = src[pidx];
				v[pos] = make_float3(s.x, s.y, s.z);
			}
			pos++;
		}
	}

	// Sort
	float3 temp;
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

	if (ix >= width || iy >= height) {
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
		float3 centerMoment = make_float3(lum * lum, lum, 0);

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
				float3 prevMoment = prevAovs[pidx].moments;

				// 積算フレーム数を１増やす.
				frame = (int)prevMoment.z + 1;

				centerMoment += prevMoment;
			}
		}

		centerMoment.z = frame;

		curAovs[idx].moments = centerMoment;
	}

	surf2Dwrite(
		make_float4(curColor, 0),
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}
#endif

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

		int curaov = getCurAovs();
		int prevaov = getPrevAovs();

		// NOTE
		// V2Cは aspect、fov、near、far から計算される.
		// 基本的にはこれらの値は変わらないので、current、prevに 関係なく利用できる.

		aten::mat4 mtxs[] = {
			m_mtxC2V,
			m_mtxV2W,
			m_mtxPrevW2V,
			m_mtxV2C,
		};

		m_mtxs.init(sizeof(aten::mat4) * AT_COUNTOF(mtxs));
		m_mtxs.writeByNum(mtxs, AT_COUNTOF(mtxs));

		temporalReprojection << <grid, block >> > (
		//temporalReprojection << <1, 1 >> > (
			m_paths.ptr(),
			m_cam.ptr(),
			m_aovNormalDepth[curaov].ptr(),
			m_aovTexclrTemporalWeight[curaov].ptr(),
			m_aovColorVariance[curaov].ptr(),
			m_aovMomentMeshid[curaov].ptr(),
			m_aovNormalDepth[prevaov].ptr(),
			m_aovTexclrTemporalWeight[prevaov].ptr(),
			m_aovColorVariance[prevaov].ptr(),
			m_aovMomentMeshid[prevaov].ptr(),
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
			m_aovTexclrTemporalWeight[curaov].ptr(),
			m_aovMomentMeshid[curaov].ptr(),
			width, height);
		checkCudaKernel(dilateWeight);

		m_mtxs.reset();
	}
}
