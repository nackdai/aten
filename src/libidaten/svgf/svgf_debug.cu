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
	else if (mode == idaten::SVGFPathTracing::AOVMode::TexColor) {
		clr = aovs[idx].texclr;
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::WireFrame) {
		bool isHitEdge = (isect.a < 1e-2) || (isect.b < 1e-2) || (1 - isect.a - isect.b < 1e-2);
		clr = isHitEdge ? make_float4(0) : make_float4(1);
	}

	surf2Dwrite(
		clr,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

__global__ void pickPixel(
	idaten::SVGFPathTracing::PickedInfo* dst,
	int ix, int iy,
	int width, int height,
	const aten::CameraParameter* __restrict__ camera,
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices)
{
	iy = height - 1 - iy;

	float s = (ix + 0.5f) / (float)(camera->width);
	float t = (iy + 0.5f) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	aten::Intersection isect;
	bool isHit = intersectBVH(&ctxt, camsample.r, &isect);

	if (isHit) {
		const auto idx = getIdx(ix, iy, width);

		const auto& path = paths[idx];
		const auto& aov = aovs[idx];

		dst->ix = ix;
		dst->iy = iy;
		dst->color = path.contrib;
		dst->normal = aten::vec3(aov.normal.x, aov.normal.y, aov.normal.z);
		dst->depth = aov.depth;
		dst->meshid = aov.meshid;
		dst->mtrlid = aov.mtrlid;
		dst->triid = isect.primid;
	}
	else {
		dst->ix = -1;
		dst->iy = -1;
	}
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

	void SVGFPathTracing::pick(
		int ix, int iy,
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
		if (m_willPicklPixel) {
			m_pick.init(1);

			auto& curAov = getCurAovs();

			pickPixel << <1, 1 >> > (
				m_pick.ptr(),
				m_pickedInfo.ix, m_pickedInfo.iy,
				width, height,
				m_cam.ptr(),
				m_paths.ptr(),
				curAov.ptr(),
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos,
				m_mtxparams.ptr());

			m_pick.readByNum(&m_pickedInfo);

			m_willPicklPixel = false;
		}
	}
}