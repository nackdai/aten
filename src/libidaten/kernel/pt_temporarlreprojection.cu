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

	*prevPos = mtxPrevV2C.apply(pos);
	*prevPos /= prevPos->w;

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

__global__ void temporalReprojection(
	const idaten::PathTracing::Path* __restrict__ paths,
	const aten::CameraParameter* __restrict__ camera,
	const idaten::PathTracingGeometryRendering::AOV* __restrict__ aovs,
	const idaten::PathTracingGeometryRendering::AOV* __restrict__ prevAOVs,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t outSurface,
	int width, int height,
	int maxSamples)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	const auto aov = aovs[idx];

	if (aov.meshid < 0) {
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

	const auto centerDepth = aten::clamp(aov.depth, camera->znear, camera->zfar);

	float3 centerNormal = aov.normal;

	// 今回のフレームのピクセルカラー.
	float4 cur = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	cur.w = 1;

#if 1
	float4 sum = make_float4(0, 0, 0, 0);
	float weight = 0.0f;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			// 前のフレームのクリップ空間座標を計算.
			aten::vec4 prevPos;
			computePrevPos(
				xx, yy,
				centerDepth,
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

				const auto pidx = getIdx(px, py, width);

				const auto prevAov = prevAOVs[pidx];
				const auto prevDepth = aten::clamp(prevAov.depth, camera->znear, camera->zfar);

				float3 prevNormal = prevAov.normal;

				static const float zThreshold = 0.05f;
				static const float nThreshold = 0.98f;

				float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
				float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
				float Wm = aov.meshid == prevAov.meshid ? 1.0f : 0.0f;

				// 前のフレームのピクセルカラーを取得.
				float4 prev;
				surf2Dread(&prev, outSurface, px * sizeof(float4), py);

				float W = Wz * Wn * Wm;
				sum += prev * W;
				weight += W;
			}
		}
	}

	if (weight > 0.0f) {
		sum /= weight;
		cur = 0.2 * cur + 0.8 * sum;
	}
#else
	// 前のフレームのクリップ空間座標を計算.
	aten::vec4 prevPos;
	computePrevPos(
		ix, iy,
		centerDepth,
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

		px = min(px, width - 1);
		py = min(py, height - 1);

		const auto pidx = getIdx(px, py, width);

		const auto prevAov = prevAOVs[pidx];
		const auto prevDepth = aten::clamp(prevAov.y, camera->znear, camera->zfar);

		// TODO
		// For trial.
		// Zの符号を正確に復元できていない.
		float3 prevNormal = make_float3(prevAov.z, prevAov.w, 0);
		prevNormal.z = sqrtf(1 - clamp(dot(prevNormal, prevNormal), 0.0f, 1.0f));

		// 前のフレームとの深度差が範囲内 && マテリアルIDが同じかどうか.
		if (abs(1 - centerDepth / prevDepth) < 0.05
			&& dot(centerNormal, prevNormal) > 0.98
			&& aov.x == prevAov.x)
		{
			// 前のフレームのピクセルカラーを取得.
			float4 prev;
			surf2Dread(&prev, outSurface, px * sizeof(float4), py);

			float2 diff = make_float2(ix - px, iy - py);
			float len = length(diff);

			if (len >= 2) {
				cur = cur;
			}
			else {
				cur = cur * 0.2 + prev * 0.8;
			}
		}
	}
#endif

	surf2Dwrite(
		cur,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);	
}

__global__ void makePathMask(
	idaten::PathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera,
	const float4* __restrict__ aovs,
	const float4* __restrict__ prevAOVs,
	const aten::mat4* __restrict__ mtxs,
	const unsigned int* sobolmatrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isKill) {
		path.isTerminate = true;
		return;
	}

	const auto aov = aovs[idx];

	const auto centerDepth = aten::clamp(aov.y, camera->znear, camera->zfar);

	// 前のフレームでのクリップ空間座標を取得.
	aten::vec4 prevPos;
	computePrevPos(
		ix, iy,
		centerDepth,
		width, height,
		&prevPos,
		mtxs);

	// [0, 1]の範囲内に入っているか.
	bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
	bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

	if (isInsideX && isInsideY) {
		// スクリーン座標に変換.
		int px = (int)(prevPos.x * width - 0.5f);
		int py = (int)(prevPos.y * height - 0.5f);

		px = min(px, width - 1);
		py = min(py, height - 1);

		const auto pidx = getIdx(px, py, width);

		const auto prevAov = prevAOVs[pidx];
		const auto prevDepth = aten::clamp(prevAov.y, camera->znear, camera->zfar);

		// 前のフレームとの深度差が範囲内 && マテリアルIDが同じかどうか.
		if (abs(1 - centerDepth / prevDepth) < 0.05
			&& aov.x == prevAov.x)
		{
			// 前のフレームのピクセルが利用できるので、パスは終了.
			path.isKill = true;
			path.isTerminate = true;
			return;
		}
	}

	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed, sobolmatrices);

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
}

__global__ void genPathTemporalReprojection(
	idaten::PathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera,
	const int* __restrict__ hitindices,
	int hitnum,
	const unsigned int* sobolmatrices)
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

	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed, sobolmatrices);

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
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		// Compute clip-view matrix.
		if (sample == 0)
		{
			m_mtxV2C.perspective(
				m_camParam.znear,
				m_camParam.zfar,
				m_camParam.vfov,
				m_camParam.aspect);

			m_mtxC2V = m_mtxV2C;
			m_mtxC2V.invert();
		}

#if 1
		PathTracingGeometryRendering::onGenPath(
			width, height,
			sample, maxSamples,
			seed,
			texVtxPos,
			texVtxNml);
#else
		if (m_isFirstRender) {
			// 最初のフレームは参照できる過去のフレームがないので、普通に処理する.
			PathTracingGeometryRendering::onGenPath(
				width, height,
				sample, maxSamples,
				seed,
				texVtxPos,
				texVtxNml);
		}
		else {
			if (sample == 0) {
				// １パス目では、AOVをレンダリングする.
				PathTracingGeometryRendering::onGenPath(
					width, height,
					sample, maxSamples,
					seed,
					texVtxPos,
					texVtxNml);
			}
			else
			{
				// 2フレーム目以降は、テンポラルリプロジェクションの隙間のみに飛ばす.

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
					m_paths.ptr(),
					m_rays.ptr(),
					width, height,
					sample, maxSamples,
					seed,
					m_cam.ptr(),
					m_aovs[cur].ptr(),
					m_aovs[prev].ptr(),
					m_mtxs.ptr(),
					m_sobolMatrices.ptr());

				m_mtxs.reset();
			}
		}
#endif
	}

	void PathTracingTemporalReprojection::onHitTest(
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
		PathTracing::onHitTest(width, height, texVtxPos);
	}

	void PathTracingTemporalReprojection::onShadeMiss(
		int width, int height,
		int bounce)
	{
		PathTracing::onShadeMiss(width, height, bounce);
	}

	void PathTracingTemporalReprojection::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		PathTracing::onShade(
			outputSurf,
			hitcount,
			width, height,
			bounce, rrBounce,
			texVtxPos, texVtxNml);
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
			//temporalReprojection << <1, 1 >> > (
				m_paths.ptr(),
				m_cam.ptr(),
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
