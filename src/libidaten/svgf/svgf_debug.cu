#include "svgf/svgf.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
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
	const float4* __restrict__ aovNormalDepth,
	const float4* __restrict__ aovTexclrTemporalWeight,
	const aten::Intersection* __restrict__ isects,
	cudaSurfaceObject_t motionDetphBuffer)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	const aten::vec3 colors[] = {
		aten::vec3(255,   0,   0),
		aten::vec3(  0, 255,   0),
		aten::vec3(  0,   0, 255),
		aten::vec3(255, 255,   0),
		aten::vec3(255,   0, 255),
		aten::vec3(  0, 255, 255),
		aten::vec3(128, 128, 128),
		aten::vec3( 86,  99, 143),
		aten::vec3( 71, 234, 126),
		aten::vec3(124,  83,  53),
	};

	const auto idx = getIdx(ix, iy, width);

	const auto& isect = isects[idx];

	float4 clr = make_float4(1);

	if (mode == idaten::SVGFPathTracing::AOVMode::Normal) {
		auto n = aovNormalDepth[idx] * 0.5f + 0.5f;
		clr = make_float4(n.x, n.y, n.z, 1);
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::Depth) {
		// TODO
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::TexColor) {
		clr = aovTexclrTemporalWeight[idx];
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::WireFrame) {
		bool isHitEdge = (isect.a < 1e-2) || (isect.b < 1e-2) || (1 - isect.a - isect.b < 1e-2);
		clr = isHitEdge ? make_float4(0) : make_float4(1);
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::BaryCentric) {
		auto c = 1 - isect.a - isect.b;
		clr = make_float4(isect.a, isect.b, c, 1);
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::Motion) {
		float4 data;
		surf2Dread(&data, motionDetphBuffer, ix * sizeof(float4), iy);

		// TODO
		float motionX = data.x;
		float motionY = data.y;

		clr = make_float4(motionX, motionY, 0, 1);
	}
	else if (mode == idaten::SVGFPathTracing::AOVMode::ObjId) {
		int objid = isect.objid;
		if (objid >= 0) {
			objid %= AT_COUNTOF(colors);
			auto c = colors[objid];

			clr = make_float4(c.x, c.y, c.z, 1);
			clr /= 255.0f;
		}
		else {
			clr = make_float4(0, 0, 0, 1);
		}
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
	const float4* __restrict__ aovNormalDepth,
	const float4* __restrict__ aovMomentMeshid,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
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
	bool isHit = intersectClosest(&ctxt, camsample.r, &isect);

	if (isHit) {
		const auto idx = getIdx(ix, iy, width);

		const auto& path = paths[idx];
		
		auto normalDepth = aovNormalDepth[idx];
		auto momentMeshid = aovMomentMeshid[idx];

		dst->ix = ix;
		dst->iy = iy;
		dst->color = path.contrib;
		dst->normal = aten::vec3(normalDepth.x, normalDepth.y, normalDepth.z);
		dst->depth = normalDepth.w;
		dst->meshid = (int)momentMeshid.w;
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

		int curaov = getCurAovs();

		CudaGLResourceMap rscmap(&m_motionDepthBuffer);
		auto gbuffer = m_motionDepthBuffer.bind();

		fillAOV << <grid, block >> > (
			outputSurf,
			m_aovMode,
			width, height,
			m_aovNormalDepth[curaov].ptr(),
			m_aovTexclrTemporalWeight[curaov].ptr(),
			m_isects.ptr(),
			gbuffer);
	}

	void SVGFPathTracing::pick(
		int ix, int iy,
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
		if (m_willPicklPixel) {
			m_pick.init(1);

			int curaov = getCurAovs();

			pickPixel << <1, 1 >> > (
				m_pick.ptr(),
				m_pickedInfo.ix, m_pickedInfo.iy,
				width, height,
				m_cam.ptr(),
				m_paths.ptr(),
				m_aovNormalDepth[curaov].ptr(),
				m_aovMomentMeshid[curaov].ptr(),
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