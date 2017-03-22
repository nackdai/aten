#pragma once

#include "types.h"
#include "math/vec3.h"

namespace aten {
	template <typename _T>
	struct TColor {
		union {
			struct {
				_T r;
				_T g;
				_T b;
				_T a;
			};
			_T c[4];
		};
	};

	class color {
	public:
		static const vec3 RGB2Y;
		static const vec3 RGB2Cb;
		static const vec3 RGB2Cr;
		static const vec3 YCbCr2R;
		static const vec3 YCbCr2G;
		static const vec3 YCbCr2B;

		static real luminance(const vec3& v)
		{
			static const vec3 lum(0.2126, 0.7152, 0.0722);
			real ret = dot(lum, v);
			return ret;
		}

		static vec3 RGBtoYCbCr(const vec3& rgb)
		{
			auto y = dot(RGB2Y, rgb);
			auto cb = dot(RGB2Cb, rgb);
			auto cr = dot(RGB2Cr, rgb);

			vec3 ycbcr(y, cb, cr);

			return std::move(ycbcr);
		}

		static real RGBtoY(const vec3& rgb)
		{
			auto y = dot(RGB2Y, rgb);
			return y;
		}

		static vec3 YCbCrtoRGB(const vec3& ycbcr)
		{
			auto r = dot(YCbCr2R, ycbcr);
			auto g = dot(YCbCr2G, ycbcr);
			auto b = dot(YCbCr2B, ycbcr);

			vec3 rgb(r, g, b);

			return std::move(rgb);
		}

		// RGB -> sRGB
		static vec3 RGBtoXYZ(const vec3& rgb)
		{
			static const vec3 _RGB2X(0.412391, 0.357584, 0.180481);
			static const vec3 _RGB2Y(0.212639, 0.715169, 0.072192);
			static const vec3 _RGB2Z(0.019331, 0.119195, 0.950532);

			auto x = dot(_RGB2X, rgb);
			auto y = dot(_RGB2Y, rgb);
			auto z = dot(_RGB2Z, rgb);

			vec3 xyz(x, y, z);

			return std::move(xyz);
		}
	};
}