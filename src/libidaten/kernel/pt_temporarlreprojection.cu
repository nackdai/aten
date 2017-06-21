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

// TODO
#define CAMERA_NEAR		(0.001f)
#define CAMERA_FAR		(10000.0f)
#define CAMERA_EPSILON	(CAMERA_NEAR / (CAMERA_FAR - CAMERA_NEAR))

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

	const aten::mat4 mtxC2V = mtxs[0];
	const aten::mat4 mtxPrevV2C = mtxs[1];

	const auto aov = aovs[idx];

	if (aov.y > CAMERA_FAR) {
		// ”wŒi‚È‚Ì‚ÅA‚»‚Ì‚Ü‚Üo—Í‚µ‚ÄI‚í‚è.
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

	aten::vec4 prevPos = mtxPrevV2C.apply(pos);
	prevPos /= prevPos.w;
	prevPos = prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]

	bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
	bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

	float4 cur = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	cur.w = 1;

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

	surf2Dwrite(
		cur,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void PathTracingTemporalReprojection::onGather(
		cudaSurfaceObject_t outputSurf,
		Path* path,
		int width, int height)
	{
		// Compute clip-view matrix.
		aten::mat4 mtxV2C;
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

			mtxV2C.m[0][0] = fW;
			mtxV2C.m[1][1] = fH;

			mtxV2C.m[2][2] = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
			mtxV2C.m[2][3] = CAMERA_NEAR * CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);

			mtxV2C.m[3][2] = -1.0f;

			mtxV2C.m[3][3] = 0.0f;

			m_mtxC2V = mtxV2C;
			m_mtxC2V.invert();
		}

		if (m_isFirstRender) {
			PathTracing::onGather(outputSurf, path, width, height);
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
				path,
				m_aovs[cur].ptr(),
				m_aovs[prev].ptr(),
				m_mtxs.ptr(),
				outputSurf,
				width, height);

			checkCudaKernel(temporarlReprojection);

			m_mtxs.reset();
		}

		m_curAOV = 1 - m_curAOV;

		m_mtxPrevV2C = mtxV2C;

		m_isFirstRender = false;
	}
}
