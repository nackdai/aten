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

inline __device__ float4 sampleTex(
	cudaSurfaceObject_t tex, 
	int ix, int iy,
	int width, int height)
{
	ix = clamp(ix, 0, width - 1);
	iy = clamp(iy, 0, height - 1);

	float4 ret;
	surf2Dread(&ret, tex, ix * sizeof(float4), iy);

	return ret;
}

__global__ void estimateVariance(
	cudaSurfaceObject_t dst,
	cudaSurfaceObject_t* aovs,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	float4 centerMoment;
	surf2Dread(
		&centerMoment,
		aovs[idaten::SVGFPathTracing::AOVType::moments],
		ix * sizeof(float4), iy);

	int frame = (int)centerMoment.w;

	centerMoment /= centerMoment.w;

	// 分散を計算.
	float var = centerMoment.x - centerMoment.y * centerMoment.y;

	if (frame < 4) {
		// 積算フレーム数が４未満 or Disoccludedされている.
		// 7x7birateral filterで輝度を計算.

		static const int radius = 3;
		static const float sigmaN = 0.005f;
		static const float sigmaD = 0.005f;
		static const float sigmaS = 8;

		float4 centerNormal;
		surf2Dread(
			&centerNormal,
			aovs[idaten::SVGFPathTracing::AOVType::normal],
			ix * sizeof(float4), iy);

		float4 vdepth;
		surf2Dread(
			&vdepth,
			aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
			ix * sizeof(float4), iy);
		float centerDepth = vdepth.x;

		float4 sum = make_float4(0, 0, 0, 0);
		float weight = 0.0f;

		for (int v = -radius; v <= radius; v++)
		{
			for (int u = -radius; u <= radius; u++)
			{
				auto moment = sampleTex(aovs[idaten::SVGFPathTracing::AOVType::moments], ix + u, iy + v, width, height);
				auto tmpNml = sampleTex(aovs[idaten::SVGFPathTracing::AOVType::normal], ix + u, iy + v, width, height);
				auto tmpDepth = sampleTex(aovs[idaten::SVGFPathTracing::AOVType::depth_meshid], ix + u, iy + v, width, height);

				moment /= moment.w;

				float4 sampleNml = tmpNml;

				float n = 1 - dot(sampleNml, centerNormal);
				float Wn = exp(-0.5f * n * n / (sigmaN * sigmaN));

				float sampleDepth = tmpDepth.x;

				float d = 1 - min(centerDepth, sampleDepth) / max(centerDepth, sampleDepth);
				float Wd = exp(-0.5f * d * d / (sigmaD * sigmaD));

				float Ws = exp(-0.5f * (u * u + v * v) / (sigmaS * sigmaS));

				float W = Ws * Wn * Wd;
				sum += moment * W;
				weight += W;
			}
		}

		if (weight > 0.0f) {
			sum /= weight;
		}

		var = sum.x - sum.y * sum.y;
	}

	// TODO
	// 分散はマイナスにならないが・・・・
	var = abs(var);

	surf2Dwrite(
		make_float4(var, var, var, var),
		aovs[idaten::SVGFPathTracing::AOVType::var],
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		make_float4(var, var, var, var),
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void SVGFPathTracing::onEstimateVariance(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		estimateVariance << <grid, block >> > (
		//estimateVariance << <1, 1 >> > (
			outputSurf,
			curaov.ptr(),
			width, height);

		checkCudaKernel(estimateVariance);
	}
}
