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

inline __device__ float4 ddx(
	int x, int y,
	int w, int h,
	cudaSurfaceObject_t s)
{
	// NOTE
	// 2x2 pixelごとに計算する.

	int leftX = x; 
	int rightX = x + 1;
	if ((x & 0x01) == 1) {
		leftX = x - 1;
		rightX = x;
	}

	rightX = min(rightX, w - 1);

	float4 left;
	float4 right;

	surf2Dread(&left, s, leftX * sizeof(float4), y);
	surf2Dread(&right, s, rightX * sizeof(float4), y);

	return right - left;
}

inline __device__ float4 ddy(
	int x, int y,
	int w, int h,
	cudaSurfaceObject_t s)
{
	// NOTE
	// 2x2 pixelごとに計算する.

	int topY = y;
	int bottomY = y + 1;
	if ((y & 0x01) == 1) {
		topY = y - 1;
		bottomY = y;
	}

	bottomY = min(bottomY, h - 1);

	float4 top;
	float4 bottom;

	surf2Dread(&top, s, x * sizeof(float4), topY);
	surf2Dread(&bottom, s, x * sizeof(float4), bottomY);

	return bottom - top;
}

inline __device__ float4 gaussFilter3x3(
	int ix, int iy,
	int w, int h,
	cudaSurfaceObject_t s)
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

			float4 tmp;
			surf2Dread(&tmp, s, xx * sizeof(float4), yy);

			sum += kernel[pos] * tmp;

			pos++;
		}
	}

	return sum;
}

template <bool isFirstIter>
__global__ void atrousFilter(
	cudaSurfaceObject_t* aovs,
	cudaSurfaceObject_t clrBuffer,
	cudaSurfaceObject_t nextClrBuffer,
	cudaSurfaceObject_t varBuffer,
	cudaSurfaceObject_t nextVarBuffer,
	int stepScale,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	float4 centerNormal;
	surf2Dread(
		&centerNormal,
		aovs[idaten::SVGFPathTracing::AOVType::normal],
		ix * sizeof(float4), iy);

	float4 centerDepthMeshId;
	surf2Dread(
		&centerDepthMeshId,
		aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
		ix * sizeof(float4), iy);

	float centerDepth = centerDepthMeshId.x;
	int centerMeshId = (int)centerDepthMeshId.y;

	float4 tmpDdzX = ddx(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::depth_meshid]);
	float4 tmpDdzY = ddy(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::depth_meshid]);
	float2 ddZ = make_float2(tmpDdzX.x, tmpDdzY.x);

	float4 centerColor;

	if (isFirstIter) {
		surf2Dread(
			&centerColor,
			aovs[idaten::SVGFPathTracing::AOVType::color],
			ix * sizeof(float4), iy);
	}
	else {
		surf2Dread(
			&centerColor,
			clrBuffer,
			ix * sizeof(float4), iy);
	}

	if (centerMeshId < 0) {
		// 背景なので、そのまま出力して終了.
		surf2Dwrite(
			centerColor,
			nextClrBuffer,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
		return;
	}

	float centerLum = AT_NAME::color::luminance(centerColor.x, centerColor.y, centerColor.z);

	// ガウスフィルタ3x3
	float4 gaussedVarLum;
	
	if (isFirstIter) {
		gaussedVarLum = gaussFilter3x3(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::var]);
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

	int idx = 0;

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

			float4 depthmeshid;
			surf2Dread(
				&depthmeshid,
				aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
				xx * sizeof(float4), yy);

			float depth = depthmeshid.x;
			int meshid = (int)depthmeshid.y;

			if (meshid != centerMeshId) {
				continue;
			}

			float4 normal;
			surf2Dread(
				&normal,
				aovs[idaten::SVGFPathTracing::AOVType::normal],
				xx * sizeof(float4), yy);

			float4 color;
			float4 variance;

			if (isFirstIter) {
				surf2Dread(
					&color,
					aovs[idaten::SVGFPathTracing::AOVType::color],
					xx * sizeof(float4), yy);
				surf2Dread(
					&variance,
					aovs[idaten::SVGFPathTracing::AOVType::var],
					xx * sizeof(float4), yy);
			}
			else {
				surf2Dread(
					&color,
					clrBuffer,
					xx * sizeof(float4), yy);
				surf2Dread(
					&variance,
					varBuffer,
					xx * sizeof(float4), yy);
			}

			float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

			float Wz = min(exp(-abs(centerDepth - depth) / (sigmaZ * dot(ddZ, p - q) + 0.000001f)), 1.0f);

			float Wn = pow(max(0.0f, dot(centerNormal, normal)), sigmaN);

			float Wl = min(exp(-abs(centerLum - lum) / (sigmaL * sqrGaussedVarLum + 0.000001f)), 1.0f);

			float Wm = meshid == centerMeshId ? 1.0f : 0.0f;

			float W = Wz * Wn * Wl * Wm;
			
			sumC += h[idx] * W * color;
			weightC += h[idx] * W;

			sumV += (h[idx] * h[idx]) * (W * W) * variance;
			weightV += h[idx] * W;

			idx++;
		}
	}

	if (weightC > 0.0) {
		sumC /= weightC;
	}
	if (weightV > 0.0) {
		sumV /= (weightV * weightV);
	}

	surf2Dwrite(
		sumC,
		nextClrBuffer,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		sumV,
		nextVarBuffer,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

__global__ void copyForNextFrame(
	cudaSurfaceObject_t srcClr,
	cudaSurfaceObject_t* aovs,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	float4 clr;
	surf2Dread(&clr, srcClr, ix * sizeof(float4), iy);
	surf2Dwrite(clr, aovs[idaten::SVGFPathTracing::AOVType::color], ix * sizeof(float4), iy, cudaBoundaryModeTrap);
}

template <bool isFirstIter>
__global__ void atrousFilterEx(
	cudaSurfaceObject_t* aovs,
	cudaSurfaceObject_t clrBuffer,
	cudaSurfaceObject_t nextClrBuffer,
	cudaSurfaceObject_t varBuffer,
	cudaSurfaceObject_t nextVarBuffer,
	int stepScale,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	float4 centerNormal;
	surf2Dread(
		&centerNormal,
		aovs[idaten::SVGFPathTracing::AOVType::normal],
		ix * sizeof(float4), iy);

	float4 centerDepthMeshId;
	surf2Dread(
		&centerDepthMeshId,
		aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
		ix * sizeof(float4), iy);

	float centerDepth = centerDepthMeshId.x;
	int centerMeshId = (int)centerDepthMeshId.y;

	float4 tmpDdzX = ddx(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::depth_meshid]);
	float4 tmpDdzY = ddy(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::depth_meshid]);
	float2 ddZ = make_float2(tmpDdzX.x, tmpDdzY.x);

	float4 centerColor;

	if (isFirstIter) {
		surf2Dread(
			&centerColor,
			aovs[idaten::SVGFPathTracing::AOVType::color],
			ix * sizeof(float4), iy);
	}
	else {
		surf2Dread(
			&centerColor,
			clrBuffer,
			ix * sizeof(float4), iy);
	}

	float centerLum = AT_NAME::color::luminance(centerColor.x, centerColor.y, centerColor.z);

	// ガウスフィルタ3x3
	float4 gaussedVarLum;

	if (isFirstIter) {
		gaussedVarLum = gaussFilter3x3(ix, iy, width, height, aovs[idaten::SVGFPathTracing::AOVType::var]);
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

	int idx = 0;

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

			float4 depthmeshid;
			surf2Dread(
				&depthmeshid,
				aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
				xx * sizeof(float4), yy);

			float depth = depthmeshid.x;
			int meshid = (int)depthmeshid.y;

			if (meshid != centerMeshId) {
				continue;
			}

			float4 normal;
			surf2Dread(
				&normal,
				aovs[idaten::SVGFPathTracing::AOVType::normal],
				xx * sizeof(float4), yy);

			float4 color;
			float4 variance;

			if (isFirstIter) {
				surf2Dread(
					&color,
					aovs[idaten::SVGFPathTracing::AOVType::color],
					xx * sizeof(float4), yy);
				surf2Dread(
					&variance,
					aovs[idaten::SVGFPathTracing::AOVType::var],
					xx * sizeof(float4), yy);
			}
			else {
				surf2Dread(
					&color,
					clrBuffer,
					xx * sizeof(float4), yy);
				surf2Dread(
					&variance,
					varBuffer,
					xx * sizeof(float4), yy);
			}

			float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

			float Wz = min(exp(-abs(centerDepth - depth) / (sigmaZ * dot(ddZ, p - q) + 0.000001f)), 1.0f);

			float Wn = pow(max(0.0f, dot(centerNormal, normal)), sigmaN);

			float Wl = min(exp(-abs(centerLum - lum) / (sigmaL * sqrGaussedVarLum + 0.000001f)), 1.0f);

			float Wm = meshid == centerMeshId ? 1.0f : 0.0f;

			float W = Wz * Wn * Wl * Wm;

			sumC += h[idx] * W * color;
			weightC += h[idx] * W;

			sumV += (h[idx] * h[idx]) * (W * W) * variance;
			weightV += h[idx] * W;

			idx++;
		}
	}

	if (weightC > 0.0) {
		sumC /= weightC;
	}
	if (weightV > 0.0) {
		sumV /= (weightV * weightV);
	}

	surf2Dwrite(
		sumC,
		nextClrBuffer,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		sumV,
		nextVarBuffer,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
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

		for (int i = 0; i < 2; i++) {
			m_atrousClrBuffer[i].map();
			m_atrousVarBuffer[i].map();
		}

		cudaSurfaceObject_t clr[] = {
			m_atrousClrBuffer[0].bind(),
			m_atrousClrBuffer[1].bind(),
		};
		cudaSurfaceObject_t var[] = {
			m_atrousVarBuffer[0].bind(),
			m_atrousVarBuffer[1].bind(),
		};

		int cur = 0;
		int next = 1;

		for (int i = 0; i < ITER; i++) {
			int stepScale = 1 << i;

			if (i == 0) {
				// First.
				atrousFilter<true> << <grid, block >> > (
					curaov.ptr(),
					clr[cur], clr[next],
					var[cur], var[next],
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);

				copyForNextFrame << <grid, block >> > (
					clr[next],
					curaov.ptr(),
					width, height);
				checkCudaKernel(copyForNextFrame);
			}
#if 1
			else if (i == ITER - 1) {
				// Final.
				atrousFilter<false> << <grid, block >> > (
				//atrousFilterEx<false> << <1, 1 >> > (
					curaov.ptr(),
					clr[cur], outputSurf,
					var[cur], var[next],
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);
			}
			else {
				atrousFilter<false> << <grid, block >> > (
					curaov.ptr(),
					clr[cur], clr[next],
					var[cur], var[next],
					stepScale,
					width, height);
				checkCudaKernel(atrousFilter);
			}
#endif

			cur = next;
			next = 1 - cur;
		}

		for (int i = 0; i < 2; i++) {
			m_atrousClrBuffer[i].unbind();
			m_atrousVarBuffer[i].unbind();

			m_atrousClrBuffer[i].unmap();
			m_atrousVarBuffer[i].unmap();
		}

	}
}
