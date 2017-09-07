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

// NOTE
// ddx, ddy
// http://mosapui.blog116.fc2.com/blog-entry-35.html
// https://www.gamedev.net/forums/topic/478820-derivative-instruction-details-ddx-ddy-or-dfdx-dfdy-etc/
// http://d.hatena.ne.jp/umonist/20110616/p1
// http://monsho.blog63.fc2.com/blog-entry-105.html

inline __device__ float ddx(
	int x, int y,
	int w, int h,
	idaten::SVGFPathTracing::AOV* aov)
{
	// NOTE
	// 2x2 pixel‚²‚Æ‚ÉŒvŽZ‚·‚é.

	int leftX = x; 
	int rightX = x + 1;
	if ((x & 0x01) == 1) {
		leftX = x - 1;
		rightX = x;
	}

	rightX = min(rightX, w - 1);

	const int idxL = getIdx(leftX, y, w);
	const int idxR = getIdx(rightX, y, w);

	float left = aov[idxL].depth;
	float right = aov[idxR].depth;

	return right - left;
}

inline __device__ float ddy(
	int x, int y,
	int w, int h,
	idaten::SVGFPathTracing::AOV* aov)
{
	// NOTE
	// 2x2 pixel‚²‚Æ‚ÉŒvŽZ‚·‚é.

	int topY = y;
	int bottomY = y + 1;
	if ((y & 0x01) == 1) {
		topY = y - 1;
		bottomY = y;
	}

	bottomY = min(bottomY, h - 1);

	int idxT = getIdx(x, topY, w);
	int idxB = getIdx(x, bottomY, w);

	float top = aov[idxT].depth;
	float bottom = aov[idxB].depth;

	return bottom - top;
}

inline __device__ float4 gaussFilter3x3(
	int ix, int iy,
	int w, int h,
	idaten::SVGFPathTracing::AOV* aov)
{
	static const float kernel[] = {
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	};

	float4 sum = make_float4(0, 0, 0, 0);

	int pos = 0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, w - 1);
			int yy = clamp(iy + y, 0, h - 1);

			int idx = getIdx(xx, yy, w);

			float4 tmp = aov[idx].var;

			sum += kernel[pos] * tmp;

			pos++;
		}
	}

	return sum;
}

inline __device__ float4 gaussFilter3x3(
	int ix, int iy,
	int w, int h,
	const float4* __restrict__ var)
{
	static const float kernel[] = {
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	};

	float4 sum = make_float4(0, 0, 0, 0);

	int pos = 0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int xx = clamp(ix + x, 0, w - 1);
			int yy = clamp(iy + y, 0, h - 1);

			int idx = getIdx(xx, yy, w);

			float4 tmp = var[idx];

			sum += kernel[pos] * tmp;

			pos++;
		}
	}

	return sum;
}

template <bool isFirstIter, bool isFinalIter>
__global__ void atrousFilter(
	cudaSurfaceObject_t dst,
	float4* tmpBuffer,
	idaten::SVGFPathTracing::AOV* aovs,
	const float4* __restrict__ clrBuffer,
	float4* nextClrBuffer,
	const float4* __restrict__ varBuffer,
	float4* nextVarBuffer,
	int stepScale,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const int idx = getIdx(ix, iy, width);

	float4 centerNormal = aovs[idx].normal;

	float centerDepth = aovs[idx].depth;
	int centerMeshId = aovs[idx].meshid;

	float tmpDdzX = ddx(ix, iy, width, height, aovs);
	float tmpDdzY = ddy(ix, iy, width, height, aovs);
	float2 ddZ = make_float2(tmpDdzX, tmpDdzY);

	float4 centerColor;

	if (isFirstIter) {
		centerColor = aovs[idx].color;
	}
	else {
		centerColor = clrBuffer[idx];
	}

	if (centerMeshId < 0) {
		// ”wŒi‚È‚Ì‚ÅA‚»‚Ì‚Ü‚Üo—Í‚µ‚ÄI—¹.
		nextClrBuffer[idx] = centerColor;

		if (isFinalIter) {
			centerColor *= aovs[idx].texclr;

			surf2Dwrite(
				centerColor,
				dst,
				ix * sizeof(float4), iy,
				cudaBoundaryModeTrap);
		}

		return;
	}

	float centerLum = AT_NAME::color::luminance(centerColor.x, centerColor.y, centerColor.z);

	// ƒKƒEƒXƒtƒBƒ‹ƒ^3x3
	float4 gaussedVarLum;
	
	if (isFirstIter) {
		gaussedVarLum = gaussFilter3x3(ix, iy, width, height, aovs);
	}
	else {
		gaussedVarLum = gaussFilter3x3(ix, iy, width, height, varBuffer);
	}

	float sqrGaussedVarLum = sqrt(gaussedVarLum.x);

	static const float sigmaZ = 1.0f;
	static const float sigmaN = 128.0f;
	static const float sigmaL = 4.0f;

	float2 p = make_float2(ix, iy);

	// NOTE
	// 5x5

	float4 sumC = make_float4(0, 0, 0, 0);
	float weightC = 0;

	float4 sumV = make_float4(0, 0, 0, 0);
	float weightV = 0;

	int pos = 0;

	static const float h[] = {
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0,  3.0 / 32.0, 3.0 / 128.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
	};

	for (int y = -2; y <= 2; y++) {
		for (int x = -2; x <= 2; x++) {
			int xx = clamp(ix + x * stepScale, 0, width - 1);
			int yy = clamp(iy + y * stepScale, 0, height - 1);

			float2 q = make_float2(xx, yy);

			const int qidx = getIdx(xx, yy, width);

			float depth = aovs[qidx].depth;
			int meshid = aovs[qidx].meshid;

			if (meshid != centerMeshId) {
				continue;
			}

			float4 normal = aovs[qidx].normal;

			float4 color;
			float4 variance;

			if (isFirstIter) {
				color = aovs[qidx].color;
				variance = aovs[qidx].var;
			}
			else {
				color = clrBuffer[qidx];
				variance = varBuffer[qidx];
			}

			float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

			float Wz = min(exp(-abs(centerDepth - depth) / (sigmaZ * abs(dot(ddZ, p - q)) + 0.000001f)), 1.0f);

			float Wn = pow(max(0.0f, dot(centerNormal, normal)), sigmaN);

			float Wl = min(exp(-abs(centerLum - lum) / (sigmaL * sqrGaussedVarLum + 0.000001f)), 1.0f);

			float Wm = meshid == centerMeshId ? 1.0f : 0.0f;

			float W = Wz * Wn * Wl * Wm;
			
			sumC += h[pos] * W * color;
			weightC += h[pos] * W;

			sumV += (h[pos] * h[pos]) * (W * W) * variance;
			weightV += h[pos] * W;

			pos++;
		}
	}

	if (weightC > 0.0) {
		sumC /= weightC;
	}
	if (weightV > 0.0) {
		sumV /= (weightV * weightV);
	}

	nextClrBuffer[idx] = sumC;
	nextVarBuffer[idx] = sumV;

	if (isFirstIter) {
		// Store color temporary.
		tmpBuffer[idx] = sumC;
	}
	
	if (isFinalIter) {
		sumC *= aovs[idx].texclr;

		surf2Dwrite(
			sumC,
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
	}
}

__global__ void copyFromBufferToAov(
	float4* src,
	idaten::SVGFPathTracing::AOV* aovs,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const int idx = getIdx(ix, iy, width);

	aovs[idx].color = src[idx];
}

namespace idaten
{
	void SVGFPathTracing::onAtrousFilter(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		static const int ITER = 5;

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		int cur = 0;
		int next = 1;

		for (int i = 0; i < ITER; i++) {
			int stepScale = 1 << i;

			if (i == 0) {
				// First.
				atrousFilter<true, false> << <grid, block >> > (
					outputSurf,
					m_tmpBuf.ptr(),
					curaov.ptr(),
					m_atrousClr[cur].ptr(), m_atrousClr[next].ptr(),
					m_atrousVar[cur].ptr(), m_atrousVar[next].ptr(),
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);
			}
#if 1
			else if (i == ITER - 1) {
				// Final.
				atrousFilter<false, true> << <grid, block >> > (
					outputSurf,
					m_tmpBuf.ptr(),
					curaov.ptr(),
					m_atrousClr[cur].ptr(), m_atrousClr[next].ptr(),
					m_atrousVar[cur].ptr(), m_atrousVar[next].ptr(),
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);
			}
			else {
				atrousFilter<false, false> << <grid, block >> > (
					outputSurf,
					m_tmpBuf.ptr(),
					curaov.ptr(),
					m_atrousClr[cur].ptr(), m_atrousClr[next].ptr(),
					m_atrousVar[cur].ptr(), m_atrousVar[next].ptr(),
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);
			}
#endif

			cur = next;
			next = 1 - cur;
		}
	}

	void SVGFPathTracing::copyFromTmpBufferToAov(int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		// Copy color from temporary buffer to AOV buffer for next temporal reprojection.
		copyFromBufferToAov << <grid, block >> > (
			m_tmpBuf.ptr(),
			curaov.ptr(),
			width, height);
		checkCudaKernel(copyFromBufferToAov);
	}
}
