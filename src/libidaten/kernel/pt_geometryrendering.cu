#include "kernel/pathtracing.h"
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

__global__ void renderAOV(
	idaten::PathTracingGeometryRendering::AOV* aovs,
	int width, int height,
	int sample, int maxSamples,
	aten::mat4 mtxW2C,
	const aten::ray* __restrict__ rays,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	const unsigned int* sobolmatrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.vtxNml = vtxNml;
		ctxt.matrices = matrices;
	}

	auto ray = rays[idx];

	aten::hitrecord rec;
	aten::Intersection isect;

	bool isHit = intersectClosest(&ctxt, ray, &isect);

	if (isHit) {
		auto obj = &ctxt.shapes[isect.objid];
		evalHitResult(&ctxt, obj, ray, &rec, &isect);

		aten::vec4 pos = aten::vec4(rec.p, 1);
		pos = mtxW2C.apply(pos);

		aovs[idx].mtrlid = isect.mtrlid;	// material id.
		aovs[idx].meshid = isect.meshid;
		aovs[idx].depth = pos.w;		// depth.

		aovs[idx].normal = make_float3(rec.normal.x, rec.normal.y, rec.normal.z);
	}
	else {
		aovs[idx].mtrlid = -1;			// material id.
		aovs[idx].meshid = -1;
		aovs[idx].depth = AT_MATH_INF;	// depth.
	}
}

enum ReferPos {
	UpperLeft,
	LowerLeft,
	UpperRight,
	LowerRight,
};

__global__ void geometryRender(
	const idaten::PathTracing::Path* __restrict__ paths,
	const idaten::PathTracingGeometryRendering::AOV* __restrict__ aovs,
	cudaSurfaceObject_t outSurface,
	int width, int height,
	int mwidth, int mheight)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	static const int ratio = 2;

	if (ix >= width && iy >= height) {
		return;
	}

	int mx = ix / (float)ratio;
	int my = iy / (float)ratio;

	// NOTE
	// +y
	// |
	// |
	// 0 ---> +x

	// NOTE
	// ul
	// +y ------- ur
	// |          |
	// |          |
	// ll ---- +x lr

	int2 pos[4] = {
		make_int2(mx, min(my + 1, mheight - 1)),						// upper-left.
		make_int2(mx, my),												// lower-left.
		make_int2(min(mx + 1, mwidth - 1), min(my + 1, mheight - 1)),	// upper-right.
		make_int2(min(mx + 1, mwidth - 1), my),							// lower-right.
	};

	// 基準点（左下）からの比率を計算.
	real u = aten::abs<int>(ix - pos[ReferPos::LowerLeft].x * ratio) / (real)ratio;
	real v = aten::abs<int>(iy - pos[ReferPos::LowerLeft].y * ratio) / (real)ratio;

	u = aten::clamp(u, AT_MATH_EPSILON, real(1));
	v = aten::clamp(v, AT_MATH_EPSILON, real(1));

	int refmidx = getIdx(ix, iy, width);
	const int mtrlIdx = aovs[refmidx].mtrlid;

	real norms[4] = {
		1 / (u * (1 - v)),
		1 / (u * v),
		1 / ((1 - u) * (1 - v)),
		1 / ((1 - u) * v),
	};

	real sumWeight = 0;

	aten::vec3 denom;
	
	for (int i = 0; i < 4; i++) {
		auto midx = getIdx(pos[i].x * ratio, pos[i].y * ratio, width);
		int refMtrlIdx = aovs[midx].mtrlid;

		int coeff = (mtrlIdx == refMtrlIdx ? 1 : 0);
		auto weight = norms[i] * coeff;;

		auto cidx = getIdx(pos[i].x, pos[i].y, mwidth);

		sumWeight += weight;
		denom += paths[cidx].contrib / (real)paths[cidx].samples * weight;
	}

	denom = denom / (sumWeight + AT_MATH_EPSILON);

	float4 data;
#if 1
	surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

	// First data.w value is 0.
	int n = data.w;
	data = n * data + make_float4(denom.x, denom.y, denom.z, 0);
	data /= (n + 1);
	data.w = n + 1;
#else
	data = make_float4(denom.x, denom.y, denom.z, 1);
#endif

	surf2Dwrite(
		data,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void PathTracingGeometryRendering::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::ShapeParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
	{
		idaten::PathTracing::update(
			gltex,
			width, height,
			camera,
			shapes,
			mtrls,
			lights,
			nodes,
			prims,
			vtxs,
			mtxs,
			texs, envmapRsc);

		// TODO
		m_aovs[0].init((width << 1) * (height << 1));
		m_aovs[1].init((width << 1) * (height << 1));
	}

	void PathTracingGeometryRendering::onGenPath(
		int width, int height,
		int sample, int maxSamples,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		idaten::PathTracing::onGenPath(
			width, height,
			sample, maxSamples,
			texVtxPos,
			texVtxNml);

		if (sample == 0) {
			renderAOVs(
				width, height,
				sample, maxSamples,
				texVtxPos,
				texVtxNml);
		}
	}

	void PathTracingGeometryRendering::renderAOVs(
		int width, int height,
		int sample, int maxSamples,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		int W = width;
		int H = height;

		aten::mat4 mtxW2V;
		mtxW2V.lookat(
			m_camParam.origin,
			m_camParam.center,
			m_camParam.up);

		aten::mat4 mtxV2C;
		mtxV2C.perspective(
			m_camParam.znear,
			m_camParam.zfar,
			m_camParam.vfov,
			m_camParam.aspect);

		aten::mat4 mtxW2C = mtxV2C * mtxW2V;

		getRenderAOVSize(W, H);

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(W + block.x - 1) / block.x,
			(H + block.y - 1) / block.y);

		auto& aovs = getCurAOVs();

		renderAOV << <grid, block >> > (
		//renderAOV << <1, 1 >> > (
			aovs.ptr(),
			W, H,
			sample, maxSamples,
			mtxW2C,
			m_rays.ptr(),
			m_shapeparam.ptr(), m_shapeparam.num(),
			m_nodetex.ptr(),
			m_primparams.ptr(),
			texVtxPos,
			texVtxNml,
			m_mtxparams.ptr(),
			m_sobolMatrices.ptr());

		checkCudaKernel(renderAOV);
	}

	void PathTracingGeometryRendering::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
		int mwidth = width;
		int mheight = height;

		width <<= 1;
		height <<= 1;

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& aovs = getCurAOVs();

		geometryRender << <grid, block >> > (
		//geometryRender << <1, 1 >> > (
			m_paths.ptr(),
			aovs.ptr(),
			outputSurf,
			width, height,
			mwidth, mheight);

		checkCudaKernel(geometryRender);
	}
}
