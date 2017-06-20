#include "kernel/pathtracing.h"
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

__global__ void renderAOV(
	int* outMtrlIds,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	aten::sampler sampler;
	sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed);

	float s = (ix + sampler.nextSample()) / (float)(camera->width);
	float t = (iy + sampler.nextSample()) / (float)(camera->height);

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

	outMtrlIds[idx] = isHit ? isect.mtrlid : -1;
}

enum ReferPos {
	UpperLeft,
	LowerLeft,
	UpperRight,
	LowerRight,
};

__global__ void geometryRender(
	const idaten::PathTracing::Path* __restrict__ paths,
	const int* __restrict__ mtrlIds,
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
	const int mtrlIdx = mtrlIds[refmidx];

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
		auto refMtrlIdx = mtrlIds[midx];

		int coeff = (mtrlIdx == refMtrlIdx ? 1 : 0);
		auto weight = norms[i] * coeff;;

		auto cidx = getIdx(pos[i].x, pos[i].y, mwidth);

		sumWeight += weight;
		denom += paths[cidx].contrib / (real)paths[cidx].samples * weight;
	}

	denom = denom / (sumWeight + AT_MATH_EPSILON);

	float4 data;
#if 0
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
		const std::vector<std::vector<aten::BVHNode>>& nodes,
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
		m_mtrlIds.init((width << 1) * (height << 1));
	}

	void PathTracingGeometryRendering::onHitTest(
		int width, int height,
		cudaTextureObject_t texVtxPos,
		int depth,
		int sample, int maxSamples,
		int seed)
	{
		if (depth == 0 && sample == 0) {
			int W = width << 1;
			int H = height << 1;

			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(
				(W + block.x - 1) / block.x,
				(H + block.y - 1) / block.y);

			renderAOV << <grid, block >> > (
			//renderAOV << <1, 1 >> > (
				m_mtrlIds.ptr(),
				W, H,
				sample, maxSamples,
				seed,
				cam.ptr(),
				shapeparam.ptr(), shapeparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				texVtxPos,
				mtxparams.ptr());

			checkCudaKernel(renderAOV);
		}

		idaten::PathTracing::onHitTest(
			width, height,
			texVtxPos,
			depth,
			sample, maxSamples,
			seed);
	}

	void PathTracingGeometryRendering::onGather(
		cudaSurfaceObject_t outputSurf,
		Path* path,
		int width, int height)
	{
		int mwidth = width;
		int mheight = height;

		width <<= 1;
		height <<= 1;

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		geometryRender << <grid, block >> > (
		//geometryRender << <1, 1 >> > (
			//paths.ptr(),
			path,
			m_mtrlIds.ptr(),
			outputSurf,
			width, height,
			mwidth, mheight);

		checkCudaKernel(geometryRender);
	}
}
