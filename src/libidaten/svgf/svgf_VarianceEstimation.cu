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

inline __device__ idaten::SVGFPathTracing::AOV* sampleAov(
	idaten::SVGFPathTracing::AOV* aovs,
	int ix, int iy,
	int width, int height)
{
	ix = clamp(ix, 0, width - 1);
	iy = clamp(iy, 0, height - 1);

	const int idx = getIdx(ix, iy, width);

	return &aovs[idx];
}

inline __device__ float3 computeViewSpace(
	int ix, int iy,
	float centerDepth,
	int width, int height,
	const aten::mat4* mtxC2V)
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

	float2 uv = make_float2(ix + 0.5, iy + 0.5);
	uv /= make_float2(width - 1, height - 1);	// [0, 1]
	uv = uv * 2.0f - 1.0f;	// [0, 1] -> [-1, 1]

	aten::vec4 pos(uv.x, uv.y, 0, 0);

	// Screen-space -> Clip-space.
	pos.x *= centerDepth;
	pos.y *= centerDepth;

	// Clip-space -> View-space
	pos = mtxC2V->apply(pos);
	pos.z = -centerDepth;
	pos.w = 1.0;

	return make_float3(pos.x, pos.y, pos.z);
}

#define IS_IN_BOUND(x, a, b)	((a) <= (x) && (x) < (b))

__global__ void varianceEstimation(
	cudaSurfaceObject_t dst,
	idaten::SVGFPathTracing::AOV* aovs,
	aten::mat4 mtxC2V,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const int idx = getIdx(ix, iy, width);

	float centerDepth = aovs[idx].depth;
	int centerMeshId = aovs[idx].meshid;

	if (centerMeshId < 0) {
		// îwåiÇ»ÇÃÇ≈ÅAï™éUÇÕÉ[Éç.
		aovs[idx].moments = make_float3(0, 0, 1);

		surf2Dwrite(
			make_float4(0),
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
	}

	auto centerViewPos = computeViewSpace(ix, iy, centerDepth, width, height, &mtxC2V);

	float3 centerMoment = aovs[idx].moments;

	int frame = (int)centerMoment.z;

	centerMoment /= centerMoment.z;

	// ï™éUÇåvéZ.
	float var = centerMoment.x - centerMoment.y * centerMoment.y;

	if (frame < 4) {
		// êœéZÉtÉåÅ[ÉÄêîÇ™ÇSñ¢ñû or DisoccludedÇ≥ÇÍÇƒÇ¢ÇÈ.
		// 7x7birateral filterÇ≈ãPìxÇåvéZ.

		static const int radius = 3;
		static const float sigmaN = 0.005f;
		static const float sigmaD = 0.005f;
		static const float sigmaS = 0.965f;

		auto centerNormal = aovs[idx].normal;

		float3 sum = make_float3(0);
		float weight = 0.0f;

		for (int v = -radius; v <= radius; v++)
		{
			for (int u = -radius; u <= radius; u++)
			{
				if (IS_IN_BOUND(ix + u, 0, width)
					&& IS_IN_BOUND(iy + v, 0, height))
				{
#if 0
					auto sampleaov = sampleAov(aovs, ix + u, iy + v, width, height);

					auto moment = sampleaov->moments;
					moment /= moment.z;

					auto sampleNml = sampleaov->normal;

					float sampleDepth = sampleaov->depth;
					int sampleMeshId = sampleaov->meshid;
#else
					int xx = clamp(ix + u, 0, width - 1);
					int yy = clamp(iy + v, 0, height - 1);

					int pidx = getIdx(xx, yy, width);
					auto v0 = aovs[pidx].v0;
					auto v3 = aovs[pidx].v3;

					float3 sampleNml = make_float3(v0.x, v0.y, v0.z);
					float sampleDepth = v0.w;
					int sampleMeshId = __float_as_int(v3.w);

					float3 moment = make_float3(v3.x, v3.y, v3.z);
					moment /= moment.z;
#endif

#if 0
					float n = 1 - dot(sampleNml, centerNormal);
					float Wn = exp(-0.5f * n * n / (sigmaN * sigmaN));

					float d = 1 - min(centerDepth, sampleDepth) / max(centerDepth, sampleDepth);
					float Wd = exp(-0.5f * d * d / (sigmaD * sigmaD));

					float Ws = exp(-0.5f * (u * u + v * v) / (sigmaS * sigmaS));
#else
					float Wn = 1.0f;
					{
						float normalCloseness = dot(sampleNml, centerNormal);
						normalCloseness = normalCloseness * normalCloseness;
						normalCloseness = normalCloseness * normalCloseness;
						float normalError = (1.0f - normalCloseness);
						Wn = max((1.0f - normalError), 0.0f);
					}

					float Wd = max(0.0f, 1.0f - abs(centerDepth - sampleDepth));

					float Ws = 1.0f;
					{
						auto sampleViewPos = computeViewSpace(ix + u, iy + v, sampleDepth, width, height, &mtxC2V);

						// Change in position in camera space.
						auto dq = centerViewPos - sampleViewPos;

						// How far away is this point from the original sample in camera space? (Max value is unbounded).
						auto dist2 = dot(dq, dq);

						// How far off the expected plane (on the perpendicular) is this point?  Max value is unbounded.
						float err = max(abs(dot(dq, sampleNml)), abs(dot(dq, centerNormal)));

						Ws = (dist2 < 0.001f)
							? 1.0
							: pow(max(0.0, 1.0 - 2.0 * err / sqrt(dist2)), 2.0);
					}
#endif

					float Wm = centerMeshId == sampleMeshId ? 1.0f : 0.0f;

					float W = Ws * Wn * Wd * Wm;
					sum += moment * W;
					weight += W;
				}
			}
		}

		if (weight > 0.0f) {
			sum /= weight;
		}

		var = sum.x - sum.y * sum.y;
	}

	// TODO
	// ï™éUÇÕÉ}ÉCÉiÉXÇ…Ç»ÇÁÇ»Ç¢Ç™ÅEÅEÅEÅE
	var = abs(var);

	aovs[idx].var = var;

	surf2Dwrite(
		make_float4(var, var, var, var),
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void SVGFPathTracing::onVarianceEstimation(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		varianceEstimation << <grid, block >> > (
		//varianceEstimation << <1, 1 >> > (
			outputSurf,
			curaov.ptr(),
			m_mtxC2V,
			width, height);

		checkCudaKernel(varianceEstimation);
	}
}
