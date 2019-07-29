#include "svgf/svgf_pt.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#define ENABLE_YCOCG

inline __device__ float3 clipAABB(
	float3 aabb_min,
	float3 aabb_max,
	float3 q)
{
	float3 center = 0.5 * (aabb_max + aabb_min);

	float3 halfsize = 0.5 * (aabb_max - aabb_min) + 0.00000001f;

	// 中心からの相対位置.
	float3 clip = q - center;

	// 相対位置の正規化.
	float3 unit = clip / halfsize;

	float3 abs_unit = make_float3(fabsf(unit.x), fabsf(unit.y), fabsf(unit.z));

	float ma_unit = max(abs_unit.x, max(abs_unit.y, abs_unit.z));

	if (ma_unit > 1.0) {
		// クリップ位置.
		return center + clip / ma_unit;
	}
	else {
		// point inside aabb
		return q;
	}
}

// https://software.intel.com/en-us/node/503873
inline __device__ float3 RGB2YCoCg(float3 c)
{
	// Y = R/4 + G/2 + B/4
	// Co = R/2 - B/2
	// Cg = -R/4 + G/2 - B/4
	return make_float3(
		c.x / 4.0 + c.y / 2.0 + c.z / 4.0,
		c.x / 2.0 - c.z / 2.0,
		-c.x / 4.0 + c.y / 2.0 - c.z / 4.0);
}

// https://software.intel.com/en-us/node/503873
inline __device__ float3 YCoCg2RGB(float3 c)
{
	// R = Y + Co - Cg
	// G = Y + Cg
	// B = Y - Co - Cg
#if 0
	return make_float3(
		clamp(c.x + c.y - c.z, 0.0f, 1.0f),
		clamp(c.x + c.z, 0.0f, 1.0f),
		clamp(c.x - c.y - c.z, 0.0f, 1.0f));
#else
	return make_float3(
		max(c.x + c.y - c.z, 0.0f),
		max(c.x + c.z, 0.0f),
		max(c.x - c.y - c.z, 0.0f));
#endif
}

inline __device__ float3 sampleColorRGB(
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs,
	int width, int height,
	int2 p)
{
	int ix = clamp(p.x, 0, width);
	int iy = clamp(p.y, 0, height);

	auto idx = getIdx(ix, iy, width);

	auto c = aovs[idx].color;

	c.x = powf(c.x, 1.0f / 2.2f);
	c.y = powf(c.y, 1.0f / 2.2f);
	c.z = powf(c.z, 1.0f / 2.2f);

	return make_float3(c.x, c.y, c.z);
}

inline __device__ float3 sampleColor(
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs,
	int width, int height,
	int2 p)
{
	auto c = sampleColorRGB(aovs, width, height, p);

#ifdef ENABLE_YCOCG
	auto ycocg = RGB2YCoCg(c);

	return ycocg;
#else
	return c;
#endif
}

inline __device__ float3 sampleColor(
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs,
	int width, int height,
	int2 p,
	float2 jitter)
{
	auto pclr = sampleColorRGB(aovs, width, height, p);
	auto jclr = sampleColorRGB(aovs, width, height, make_int2(p.x - 1, p.y - 1));

	auto f = sqrtf(dot(jitter, jitter)) / sqrtf(2.0f);

	auto ret = lerp(jclr, pclr, f);

#ifdef ENABLE_YCOCG
	ret = RGB2YCoCg(ret);
#endif

	return ret;
}

inline __device__ float3 min(float3 a, float3 b)
{
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline __device__ float3 max(float3 a, float3 b)
{
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline __device__ float4 abs(float4 v)
{
	v.x = fabsf(v.x);
	v.y = fabsf(v.y);
	v.z = fabsf(v.z);
	v.w = fabsf(v.w);
	return v;
}

inline __device__ float3 clipColor(
	float3 clr0, float3 clr1,
	int ix, int iy,
	int width, int height,
	float2 jitter,
	const idaten::SVGFPathTracing::AOV* __restrict__ aovs)
{
	int2 du = make_int2(1, 0);
	int2 dv = make_int2(0, 1);

	int2 uv = make_int2(ix, iy);

	float3 ctl = sampleColor(aovs, width, height, uv - dv - du, jitter);
	float3 ctc = sampleColor(aovs, width, height, uv - dv, jitter);
	float3 ctr = sampleColor(aovs, width, height, uv - dv + du, jitter);
	float3 cml = sampleColor(aovs, width, height, uv - du, jitter);
	float3 cmc = sampleColor(aovs, width, height, uv, jitter);
	float3 cmr = sampleColor(aovs, width, height, uv + du, jitter);
	float3 cbl = sampleColor(aovs, width, height, uv + dv - du, jitter);
	float3 cbc = sampleColor(aovs, width, height, uv + dv, jitter);
	float3 cbr = sampleColor(aovs, width, height, uv + dv + du, jitter);

	float3 cmin = min(ctl, min(ctc, min(ctr, min(cml, min(cmc, min(cmr, min(cbl, min(cbc, cbr))))))));
	float3 cmax = max(ctl, max(ctc, max(ctr, max(cml, max(cmc, max(cmr, max(cbl, max(cbc, cbr))))))));

	float3 cavg = (ctl + ctc + ctr + cml + cmc + cmr + cbl + cbc + cbr) / 9.0;

	float3 cmin5 = min(ctc, min(cml, min(cmc, min(cmr, cbc))));
	float3 cmax5 = max(ctc, max(cml, max(cmc, max(cmr, cbc))));
	float3 cavg5 = (ctc + cml + cmc + cmr + cbc) / 5.0;
	cmin = 0.5 * (cmin + cmin5);
	cmax = 0.5 * (cmax + cmax5);
	cavg = 0.5 * (cavg + cavg5);

#ifdef ENABLE_YCOCG
	auto ex = 0.25 * 0.5 * (cmax.x - cmin.x);
	float2 chroma_extent = make_float2(ex, ex);
	float2 chroma_center = make_float2(clr0.y, clr0.z);
	
	cmin.y = chroma_center.x - chroma_extent.x;
	cmin.z = chroma_center.y - chroma_extent.y;

	cmax.y = chroma_center.x + chroma_extent.x;
	cmax.z = chroma_center.y + chroma_extent.y;

	cavg.y = chroma_center.x;
	cavg.z = chroma_center.y;
#endif

	auto ret = clipAABB(cmin, cmax, clr1);

	return ret;
}

inline __device__ float4 PDnrand4(float2 n)
{
	return fracf(sin(dot(n, make_float2(12.9898f, 78.233f))) * make_float4(43758.5453f, 28001.8384f, 50849.4141f, 12996.89f));
}

inline __device__ float4 PDsrand4(float2 n)
{
	return PDnrand4(n) * 2 - 1;
}

// TODO
// temporal reporjectionのものと統一するべき.
inline __device__ void computePrevScreenPos(
	int ix, int iy,
	float centerDepth,
	int width, int height,
	aten::vec4* prevPos,
	const aten::mat4 mtxC2V,
	const aten::mat4 mtxPrevV2C)
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

	// Reproject previous screen position.
	*prevPos = mtxPrevV2C.apply(pos);
	*prevPos /= prevPos->w;

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

template <bool showDiff>
__global__ void temporalAA(
	cudaSurfaceObject_t dst,
	idaten::SVGFPathTracing::Path* paths,
	const idaten::SVGFPathTracing::AOV* __restrict__ curAovs,
	const idaten::SVGFPathTracing::AOV* __restrict__ prevAovs,
	const aten::mat4 mtxC2V,
	const aten::mat4 mtxPrevV2C,
	float2 jitter,
	float sinTime,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const int idx = getIdx(ix, iy, width);

	auto p = make_int2(ix, iy);

	auto* sampler = &paths[idx].sampler;

	auto tmp = sampleColorRGB(curAovs, width, height, p);

	auto clr0 = sampleColor(curAovs, width, height, p, jitter);

	auto centerDepth = curAovs[idx].depth;

	aten::vec4 prevPos;
	computePrevScreenPos(
		ix, iy,
		centerDepth,
		width, height,
		&prevPos,
		mtxC2V, mtxPrevV2C);

	// [0, 1]の範囲内に入っているか.
	bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
	bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

	if (isInsideX && isInsideY) {
		// 前のフレームのスクリーン座標.
		int px = (int)(prevPos.x * width - 0.5f);
		int py = (int)(prevPos.y * height - 0.5f);

		auto clr1 = sampleColor(prevAovs, width, height, make_int2(px, py));

		clr1 = clipColor(clr0, clr1, ix, iy, width, height, jitter, curAovs);

#ifdef ENABLE_YCOCG
		float lum0 = clr0.x;
		float lum1 = clr1.x;
#else
		float lum0 = AT_NAME::color::luminance(clr0.x, clr0.y, clr0.z);
		float lum1 = AT_NAME::color::luminance(clr1.x, clr1.y, clr1.z);
#endif

		float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2f));
		float unbiased_weight = 1.0 - unbiased_diff;
		float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
		float k_feedback = lerp(0.3, 0.6, unbiased_weight_sqr);

		auto c = lerp(clr0, clr1, k_feedback);

#ifdef ENABLE_YCOCG
		c = YCoCg2RGB(c);
#endif

		auto noise4 = abs(PDsrand4(make_float2(ix, iy) + sinTime + 0.6959174f) / 510.0f);
		noise4.w = 0;

		auto f = make_float4(c.x, c.y, c.z, 1) + noise4;

		if (showDiff) {
			f = make_float4(abs(f.x - tmp.x), abs(f.y - tmp.y), abs(f.z - tmp.z), 1);
		}

		surf2Dwrite(
			f,
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
	}
	else if (showDiff) {
		surf2Dwrite(
			make_float4(0, 0, 0, 1),
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
	}
}

namespace idaten
{
	void SVGFPathTracing::onTAA(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		if (isFirstFrame()) {
			return;
		}

		if (isEnableTAA()) {
			auto& curaov = getCurAovs();
			auto& prevaov = getPrevAovs();

			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(
				(width + block.x - 1) / block.x,
				(height + block.y - 1) / block.y);

			float sintime = aten::abs(aten::sin((m_frame & 0xf) * AT_MATH_PI));

			// http://en.wikipedia.org/wiki/Halton_sequence
			const static float2 offset[8] = {
				make_float2( 1.0f / 2.0f, 1.0f / 3.0f ),
				make_float2( 1.0f / 4.0f, 2.0f / 3.0f ),
				make_float2( 3.0f / 4.0f, 1.0f / 9.0f ),
				make_float2( 1.0f / 8.0f, 4.0f / 9.0f ),
				make_float2( 5.0f / 8.0f, 7.0f / 9.0f ),
				make_float2( 3.0f / 8.0f, 2.0f / 9.0f ),
				make_float2( 7.0f / 8.0f, 5.0f / 9.0f ),
				make_float2( 1.0f / 16.0f, 8.0f / 9.0f ),
			};

			auto frame = (m_frame + 1) & 0x7;

			if (canShowTAADiff()) {
				temporalAA<true> << <grid, block >> > (
					outputSurf,
					m_paths.ptr(),
					curaov.ptr(),
					prevaov.ptr(),
					m_mtxC2V,
					m_mtxPrevV2C,
					offset[frame],
					sintime,
					width, height);
			}
			else {
				temporalAA<false> << <grid, block >> > (
				//temporalAA<false> << <1, 1 >> > (
					outputSurf,
					m_paths.ptr(),
					curaov.ptr(),
					prevaov.ptr(),
					m_mtxC2V,
					m_mtxPrevV2C,
					offset[frame],
					sintime,
					width, height);
			}

			checkCudaKernel(temporalAA);
		}
	}
}