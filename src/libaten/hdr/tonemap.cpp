#include <tuple>
#include "hdr/tonemap.h"

namespace aten
{
	static const vec3 RGB2Y(CONST_REAL(0.29900), CONST_REAL(0.58700), CONST_REAL(0.11400));
	static const vec3 RGB2Cb(CONST_REAL(-0.16874), CONST_REAL(-0.33126), CONST_REAL(0.50000));
	static const vec3 RGB2Cr(CONST_REAL(0.50000), CONST_REAL(-0.41869), CONST_REAL(-0.08131));
	static const vec3 YCbCr2R(CONST_REAL(1.00000), CONST_REAL(0.00000), CONST_REAL(1.40200));
	static const vec3 YCbCr2G(CONST_REAL(1.00000), CONST_REAL(-0.34414), CONST_REAL(-0.71414));
	static const vec3 YCbCr2B(CONST_REAL(1.00000), CONST_REAL(1.77200), CONST_REAL(0.00000));

	// NOTE
	// Reinherd ‚Ì•½‹Ï‹P“xŒvŽZ.
	// Lavg = exp(1/N * ƒ°log(ƒÂ + L(x, y)))

	// NOTE
	// HDR
	// http://t-pot.com/program/123_ToneMapping/index.html

	std::tuple<real, real> computeAvgAndMaxLum(
		int width, int height,
		const vec3* src)
	{
		real sumY = CONST_REAL(0.0);
		real maxLum = CONST_REAL(0.0);

		int cnt = 0;

		for(int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;

				const vec3& s = src[pos];

				vec3 col(
					aten::sqrt(s.r),
					aten::sqrt(s.g),
					aten::sqrt(s.b));

				real lum = dot(RGB2Y, col);

				if (lum > CONST_REAL(0.0)) {
					sumY += aten::log(lum);

					if (lum > maxLum) {
						maxLum = lum;
					}

					cnt++;
				}
			}
		}

		sumY /= cnt;

		std::tuple<real, real> result = std::make_tuple(sumY, maxLum);
		return result;
	}

	void Tonemap::doTonemap(
		int width, int height,
		const vec3* src,
		color* dst)
	{
		auto result = computeAvgAndMaxLum(
			width, height,
			src);

		auto lum = std::get<0>(result);
		auto maxlum = std::get<1>(result);

		static const real middleGrey = CONST_REAL(0.18);

		const real coeff = middleGrey / aten::exp(lum);
		const real l_max = coeff * maxlum;

		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				int pos = h * width + w;

				const vec3& s = src[pos];
				color& d = dst[pos];

				vec3 col(
					aten::sqrt(s.r),
					aten::sqrt(s.g),
					aten::sqrt(s.b));

				real y = dot(RGB2Y, col);
				real cb = dot(RGB2Cb, col);
				real cr = dot(RGB2Cr, col);

				y = coeff * y;
				y = y * (1.0f + y / (l_max * l_max)) / (1.0f + y);

				vec3 ycbcr(y, cb, cr);

				real r = dot(YCbCr2R, ycbcr);
				real g = dot(YCbCr2G, ycbcr);
				real b = dot(YCbCr2B, ycbcr);

				int ir = int(255.9f * r);
				int ig = int(255.9f * g);
				int ib = int(255.9f * b);

				d.r = aten::clamp(ir, 0, 255);
				d.g = aten::clamp(ig, 0, 255);
				d.b = aten::clamp(ib, 0, 255);
				d.a = 255;
			}
		}
	}
}