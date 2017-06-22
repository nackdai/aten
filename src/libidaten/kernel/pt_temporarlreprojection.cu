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

//#define ENABLE_DEBUG

// TODO
#define CAMERA_NEAR		(0.001f)
#define CAMERA_FAR		(10000.0f)
#define CAMERA_EPSILON	(CAMERA_NEAR / (CAMERA_FAR - CAMERA_NEAR))

inline __device__ void computePrevPos(
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
	// mtxV2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, -Zview)
	//  Wclip = -Zview = -depth
	// Xscr = Xclip / Wclip = Xclip / -Zview = Xclip / -depth
	// Yscr = Yclip / Wclip = Yclip / -Zview = Yclip / -depth
	//
	// Xscr * -depth = Xclip
	// Xview = mtxC2V * Xclip

	const aten::mat4 mtxC2V = mtxs[0];
	const aten::mat4 mtxPrevV2C = mtxs[1];

	float2 uv = make_float2(ix, iy);
	uv /= make_float2(width - 1, height - 1);	// [0 - 1]
	uv = uv * 2.0f - 1.0f;	// [0 - 1] -> [-1, 1]

	aten::vec4 pos(uv.x, uv.y, 0, 1);

	// Screen-space -> Clip-space.
	pos.x *= -centerDepth;
	pos.y *= -centerDepth;

	// Clip-space -> View-space
	pos = mtxC2V.apply(pos);
	pos.z = centerDepth;

	*prevPos = mtxPrevV2C.apply(pos);
	*prevPos /= prevPos->w;

	// TODO
	// 誤差対策.
	if (prevPos->x < -1) {
		prevPos->x += 0.1f / width;
	}
	else if (prevPos->x > 1) {
		prevPos->x -= 0.1f / width;
	}

	if (prevPos->y < -1) {
		prevPos->y += 0.1f / height;
	}
	else if (prevPos->y > 1) {
		prevPos->y -= 0.1f / height;
	}

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

#ifdef ENABLE_DEBUG
// For debug.
__global__ void temporalReprojection(
	const idaten::PathTracing::Path* __restrict__ paths,
	const float4* __restrict__ aovs,
	const float4* __restrict__ prevAOVs,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t outSurface,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	float4 cur = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	cur.w = 1;

	surf2Dwrite(
		cur,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}
#else
__global__ void temporalReprojection(
	const idaten::PathTracing::Path* __restrict__ paths,
	const float4* __restrict__ aovs,
	const float4* __restrict__ prevAOVs,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t outSurface,
	int width, int height,
	int maxSamples)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	const auto aov = aovs[idx];

	if (aov.y > CAMERA_FAR) {
		// 背景なので、そのまま出力して終わり.
		float4 clr = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
		clr.w = 1;

		surf2Dwrite(
			clr,
			outSurface,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		return;
	}

	const auto centerDepth = aten::clamp(aov.y, CAMERA_NEAR, CAMERA_FAR);

	aten::vec4 prevPos;
	computePrevPos(
		ix, iy,
		centerDepth,
		width, height,
		&prevPos,
		mtxs);

	bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
	bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

	float4 cur = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	cur.w = 1;

#if 0
	if (isInsideX && isInsideY) {
		int px = (int)(prevPos.x * (width - 1) + 0.5f);
		int py = (int)(prevPos.y * (height - 1) + 0.5f);

		px = min(px, width - 1);
		py = min(py, height - 1);

		const auto pidx = getIdx(px, py, width);

		const auto prevAov = prevAOVs[pidx];
		const auto prevDepth = aten::clamp(prevAov.y, CAMERA_NEAR, CAMERA_FAR);

		if (abs(centerDepth - prevDepth) <= 0.1f
			&& aov.x == prevAov.x)
		{
			float4 prev;
			surf2Dread(&prev, outSurface, px * sizeof(float4), py);

			int n = (int)prev.w;

			// TODO
			n = min(1, n);

			cur = prev * n + cur * path.samples;
			cur /= (float)(n + path.samples);
			cur.w = n + path.samples;
		}
	}
#endif

	surf2Dwrite(
		cur,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}
#endif

__global__ void makePathMask(
	const float4* __restrict__ aovs,
	const float4* __restrict__ prevAOVs,
	const aten::mat4* __restrict__ mtxs,
	int* hitbools,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	hitbools[idx] = 1;

	const auto aov = aovs[idx];

	const auto centerDepth = aten::clamp(aov.y, CAMERA_NEAR, CAMERA_FAR);

	aten::vec4 prevPos;
	computePrevPos(
		ix, iy,
		centerDepth,
		width, height,
		&prevPos,
		mtxs);

	bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
	bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

	if (isInsideX && isInsideY) {
		int px = (int)(prevPos.x * (width - 1) + 0.5f);
		int py = (int)(prevPos.y * (height - 1) + 0.5f);

		px = min(px, width - 1);
		py = min(py, height - 1);

		const auto pidx = getIdx(px, py, width);

		const auto prevAov = prevAOVs[pidx];
		const auto prevDepth = aten::clamp(prevAov.y, CAMERA_NEAR, CAMERA_FAR);

		if (abs(centerDepth - prevDepth) <= 0.1f
			&& aov.x == prevAov.x)
		{
			hitbools[idx] = 0;
		}
		else {
			hitbools[idx] = 1;
		}
	}
	else {
		hitbools[idx] = 1;
	}
}

__global__ void genPathTemporalReprojection(
	idaten::PathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera,
	const int* __restrict__ hitindices,
	int hitnum)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= width * height) {
		return;
	}

	paths[idx].isHit = false;
	paths[idx].isTerminate = true;

	if (idx >= hitnum) {
		return;
	}

	idx = hitindices[idx];

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isKill) {
		path.isTerminate = true;
		return;
	}

	int ix = idx % width;
	int iy = idx / height;

	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed);

	float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
	float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	rays[idx] = camsample.r;

	path.throughput = aten::vec3(1);
	path.pdfb = 0.0f;
	path.isTerminate = false;
	path.isSingular = false;

	path.samples += 1;

#ifdef ENABLE_DEBUG
	path.contrib = aten::vec3(1, 0, 0);
#endif
}

namespace idaten
{
	void PathTracingTemporalReprojection::update(
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
		idaten::PathTracingGeometryRendering::update(
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

		m_hitboolsTemporal.init(width * height);
		m_hitidxTemporal.init(width * height);
	}

	void PathTracingTemporalReprojection::onGenPath(
		int width, int height,
		int sample, int maxSamples,
		int seed,
		cudaTextureObject_t texVtxPos)
	{
		// Compute clip-view matrix.
		if (sample == 0)
		{
			/*
			* D3DXMatrixPerspectiveFovRH
			*
			* xScale     0          0              0
			* 0        yScale       0              0
			* 0        0        zf/(zn-zf)   zn*zf/(zn-zf)
			* 0        0            -1             0
			* where:
			* yScale = cot(fovY/2)
			*
			* xScale = yScale / aspect ratio
			*/

			// Use Vertical FOV
			const real fH = 1 / aten::tan(Deg2Rad(m_camParam.vfov) * 0.5f);
			const real fW = fH / m_camParam.aspect;

			m_mtxV2C.m[0][0] = fW;
			m_mtxV2C.m[1][1] = fH;

			m_mtxV2C.m[2][2] = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
			m_mtxV2C.m[2][3] = CAMERA_NEAR * CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);

			m_mtxV2C.m[3][2] = -1.0f;

			m_mtxV2C.m[3][3] = 0.0f;

			m_mtxC2V = m_mtxV2C;
			m_mtxC2V.invert();
		}

#if 1
		if (m_isFirstRender) {
			PathTracingGeometryRendering::onGenPath(
				width, height,
				sample, maxSamples,
				seed,
				texVtxPos);
		}
		else {
			if (sample == 0) {
				// まずは全体にレイを１つ飛ばす.
				PathTracingGeometryRendering::onGenPath(
					width, height,
					sample, maxSamples,
					seed,
					texVtxPos);
			}
			else {
				// 2レイ目以降は、テンポラルリプロジェクションの隙間のみに飛ばす.

				int cur = m_curAOV;
				int prev = 1 - cur;

				aten::mat4 mtxs[2] = {
					m_mtxC2V,
					m_mtxPrevV2C,
				};

				m_mtxs.init(sizeof(aten::mat4) * AT_COUNTOF(mtxs));
				m_mtxs.writeByNum(mtxs, AT_COUNTOF(mtxs));

				dim3 block(BLOCK_SIZE, BLOCK_SIZE);
				dim3 grid(
					(width + block.x - 1) / block.x,
					(height + block.y - 1) / block.y);

				makePathMask << <grid, block >> > (
					m_aovs[cur].ptr(),
					m_aovs[prev].ptr(),
					m_mtxs.ptr(),
					m_hitboolsTemporal.ptr(),
					width, height);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidxTemporal,
					m_hitboolsTemporal,
					&hitcount);

				dim3 blockPerGrid((width * height + 128 - 1) / 128);
				dim3 threadPerBlock(128);

				genPathTemporalReprojection << <blockPerGrid, threadPerBlock >> > (
					paths.ptr(),
					rays.ptr(),
					width, height,
					sample, maxSamples,
					seed,
					cam.ptr(),
					m_hitidxTemporal.ptr(),
					hitcount);

				m_mtxs.reset();
			}
		}
#else
		PathTracingGeometryRendering::onGenPath(
			width, height,
			sample, maxSamples,
			seed,
			texVtxPos);
#endif
	}

	void PathTracingTemporalReprojection::onHitTest(
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
#ifndef ENABLE_DEBUG
		PathTracing::onHitTest(width, height, texVtxPos);
#endif
	}

	void PathTracingTemporalReprojection::onShadeMiss(
		int width, int height,
		int depth)
	{
#ifndef ENABLE_DEBUG
		PathTracing::onShadeMiss(width, height, depth);
#endif
	}

	void PathTracingTemporalReprojection::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int depth, int rrDepth,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
#ifndef ENABLE_DEBUG
		PathTracing::onShade(
			outputSurf,
			hitcount,
			depth, rrDepth,
			texVtxPos, texVtxNml);
#endif
	}

	void PathTracingTemporalReprojection::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
		if (m_isFirstRender) {
			PathTracing::onGather(outputSurf, width, height, maxSamples);
		}
		else {
			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(
				(width + block.x - 1) / block.x,
				(height + block.y - 1) / block.y);

			int cur = m_curAOV;
			int prev = 1 - cur;

			aten::mat4 mtxs[2] = {
				m_mtxC2V,
				m_mtxPrevV2C,
			};

			m_mtxs.init(sizeof(aten::mat4) * AT_COUNTOF(mtxs));
			m_mtxs.writeByNum(mtxs, AT_COUNTOF(mtxs));

			temporalReprojection << <grid, block >> > (
			//temporarlReprojection << <1, 1 >> > (
				paths.ptr(),
				m_aovs[cur].ptr(),
				m_aovs[prev].ptr(),
				m_mtxs.ptr(),
				outputSurf,
				width, height,
				maxSamples);

			checkCudaKernel(temporarlReprojection);

			m_mtxs.reset();
		}

		m_curAOV = 1 - m_curAOV;

		m_mtxPrevV2C = m_mtxV2C;

		m_isFirstRender = false;
	}
}
