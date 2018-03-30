#pragma once

#include "types.h"
#include "math/vec3.h"

#define AT_COLOR_RGBA(r, g, b, a) ((b) | ((g) << 8) | ((r) << 16) | ((a) << 24))
#define AT_COLOR_NORMALIZE(c)	((c) / real(255))

namespace AT_NAME {
	template <typename _T, int N>
	struct TColor {
		_T c[N];

		_T& r()
		{
			return c[0];
		}
		_T& g()
		{
			return c[1];
		}
		_T& b()
		{
			return c[2];
		}
	};

	template <typename _T>
	struct TColor<_T, 4> {
		_T c[4];

		_T& r()
		{
			return c[0];
		}
		_T& g()
		{
			return c[1];
		}
		_T& b()
		{
			return c[2];
		}
		_T& a()
		{
			return c[3];
		}
	};

	class color {
	public:
		static const aten::vec3 RGB2Y;
		static const aten::vec3 RGB2Cb;
		static const aten::vec3 RGB2Cr;
		static const aten::vec3 YCbCr2R;
		static const aten::vec3 YCbCr2G;
		static const aten::vec3 YCbCr2B;

		static inline AT_DEVICE_API real luminance(const aten::vec3& v)
		{
			real ret = dot(aten::vec3(real(0.2126), real(0.7152), real(0.0722)), v);
			return ret;
		}

		static inline AT_DEVICE_API real luminance(real r, real g, real b)
		{
			real ret = real(0.2126) * r + real(0.7152) * g + real(0.0722) * b;
			return ret;
		}

		static aten::vec3 RGBtoYCbCr(const aten::vec3& rgb)
		{
			auto y = dot(RGB2Y, rgb);
			auto cb = dot(RGB2Cb, rgb);
			auto cr = dot(RGB2Cr, rgb);

			aten::vec3 ycbcr = aten::vec3(y, cb, cr);

			return std::move(ycbcr);
		}

		static real RGBtoY(const aten::vec3& rgb)
		{
			auto y = dot(RGB2Y, rgb);
			return y;
		}

		static aten::vec3 YCbCrtoRGB(const aten::vec3& ycbcr)
		{
			auto r = dot(YCbCr2R, ycbcr);
			auto g = dot(YCbCr2G, ycbcr);
			auto b = dot(YCbCr2B, ycbcr);

			aten::vec3 rgb = aten::vec3(r, g, b);

			return std::move(rgb);
		}

		// RGB -> sRGB
		static aten::vec3 RGBtoXYZ(const aten::vec3& rgb)
		{
			static const aten::vec3 _RGB2X = aten::vec3(real(0.412391), real(0.357584), real(0.180481));
			static const aten::vec3 _RGB2Y = aten::vec3(real(0.212639), real(0.715169), real(0.072192));
			static const aten::vec3 _RGB2Z = aten::vec3(real(0.019331), real(0.119195), real(0.950532));

			auto x = dot(_RGB2X, rgb);
			auto y = dot(_RGB2Y, rgb);
			auto z = dot(_RGB2Z, rgb);

			aten::vec3 xyz = aten::vec3(x, y, z);

			return std::move(xyz);
		}
	};
}