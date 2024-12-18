#pragma once

#include <array>

#include "types.h"
#include "math/vec3.h"

#define AT_COLOR_RGBA(r, g, b, a) ((b) | ((g) << 8) | ((r) << 16) | ((a) << 24))
#define AT_COLOR_NORMALIZE(c)    ((c) / float(255))

namespace AT_NAME {
    template <class _T, int32_t N>
    struct TColor {
        std::array<_T, N> c;

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
        static constexpr size_t BPP = sizeof(_T) * N;
    };

    template <class _T>
    struct TColor<_T, 4> {
        std::array<_T, 4> c;

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

        static inline AT_HOST_DEVICE_API float luminance(const aten::vec3& v)
        {
            float ret = luminance(v.r, v.g, v.b);
            return ret;
        }

        static inline AT_HOST_DEVICE_API float luminance(float r, float g, float b)
        {
            float ret = float(0.2126) * r + float(0.7152) * g + float(0.0722) * b;
            return ret;
        }

        static aten::vec3 RGBtoYCbCr(const aten::vec3& rgb)
        {
            auto y = dot(RGB2Y, rgb);
            auto cb = dot(RGB2Cb, rgb);
            auto cr = dot(RGB2Cr, rgb);

            aten::vec3 ycbcr = aten::vec3(y, cb, cr);

            return ycbcr;
        }

        static float RGBtoY(const aten::vec3& rgb)
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

            return rgb;
        }

        // RGB -> sRGB
        static AT_DEVICE_API aten::vec3 RGBtoXYZ(const aten::vec3& rgb)
        {
            // NOTE:
            // CUDA compiler doesn't allow dynamic static initialization during compile.
            const aten::vec3 _RGB2X = aten::vec3(float(0.412391), float(0.357584), float(0.180481));
            const aten::vec3 _RGB2Y = aten::vec3(float(0.212639), float(0.715169), float(0.072192));
            const aten::vec3 _RGB2Z = aten::vec3(float(0.019331), float(0.119195), float(0.950532));

            auto x = dot(_RGB2X, rgb);
            auto y = dot(_RGB2Y, rgb);
            auto z = dot(_RGB2Z, rgb);

            aten::vec3 xyz = aten::vec3(x, y, z);

            return xyz;
        }
    };
}
