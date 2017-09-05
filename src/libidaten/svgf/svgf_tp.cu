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

	float4 centerNormal = curAovs[idx].normal;

	float4 sum = make_float4(0, 0, 0, 0);
	float weight = 0.0f;

	float4 prevNormal;

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

				prevNormal = prevAovs[pidx].normal;

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
		curColor = 0.2 * curColor + 0.8 * sum;
	}

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

			prevNormal = prevAovs[pidx].normal;

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
			width, height);

		checkCudaKernel(temporalReprojection);

		m_mtxs.reset();
	}
}
