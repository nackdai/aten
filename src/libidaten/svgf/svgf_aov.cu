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

__global__ void fillAOV(
	cudaSurfaceObject_t dst,
	idaten::SVGFPathTracing::AOVMode mode,
	int width, int height,
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs,
	const aten::Intersection* __restrict__ isects)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto& isect = isects[idx];

	float4 clr = make_float4(1);

	if (mode == idaten::SVGFPathTracing::AOVMode::Normal) {
		clr = aovs[idx].normal * 0.5f + 0.5f;
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::Depth) {
		// TODO
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::WireFrame) {
		bool hit = (isect.a < 1e-2) || (isect.b < 1e-2) || (1 - isect.a - isect.b < 1e-2);
		clr = hit ? make_float4(0) : make_float4(1);
	}

	surf2Dwrite(
		clr,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void SVGFPathTracing::onFillAOV(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		fillAOV << <grid, block >> > (
			outputSurf,
			m_aovMode,
			width, height,
			curaov.ptr(),
			m_isects.ptr());
	}
}