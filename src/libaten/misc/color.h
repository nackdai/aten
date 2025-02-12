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
        static inline AT_HOST_DEVICE_API float luminance(const aten::vec3& v)
        {
            float ret = luminance(v.r, v.g, v.b);
            return ret;
        }

        static inline AT_HOST_DEVICE_API float luminance(float r, float g, float b)
        {
#if 0
            // NOTE:
            // ITU-R BT.601.
            float ret = 0.299F * r + 0.587F * g + 0.114F * b;
#else
            // Y in XYZ from sRGB.
            // This assumes sRGB is linear.
            float ret = 0.212639F * r + 0.71517F * g + 0.0721926F * b;
#endif
            return ret;
        }

        static AT_DEVICE_API aten::vec3 sRGBtoXYZ(const aten::vec3& srgb)
        {
            // NOTE:
            // CUDA compiler doesn't allow dynamic static initialization during compile.
            const aten::vec3 _sRGB2X = aten::vec3(0.412391F, 0.357585F, 0.180482F);
            const aten::vec3 _sRGB2Y = aten::vec3(0.212639F, 0.71517F, 0.0721926F);
            const aten::vec3 _sRGB2Z = aten::vec3(0.0193308F, 0.119195F, 0.950536F);

            auto x = dot(_sRGB2X, srgb);
            auto y = dot(_sRGB2Y, srgb);
            auto z = dot(_sRGB2Z, srgb);

            return aten::vec3(x, y, z);
        }

        static AT_DEVICE_API aten::vec3 XYZtosRGB(const aten::vec3& xyz)
        {
            // NOTE:
            // CUDA compiler doesn't allow dynamic static initialization during compile.
            const aten::vec3 _XYZ2R = aten::vec3(3.2406F, -0.9689F, 0.0557F);
            const aten::vec3 _XYZ2G = aten::vec3(-1.5372F, 1.8758F, -0.2040F);
            const aten::vec3 _XYZ2B = aten::vec3(-0.4986F, 0.0415F, 1.0570F);

            auto r = dot(_XYZ2R, xyz);
            auto g = dot(_XYZ2G, xyz);
            auto b = dot(_XYZ2B, xyz);

            return aten::vec3(r, g, b);
        }
    };

    class ColorEncoder {
    public:
        ColorEncoder() = default;
        virtual ~ColorEncoder() = default;

        ColorEncoder(const ColorEncoder&) = delete;
        ColorEncoder(ColorEncoder&&) = delete;
        ColorEncoder& operator=(const ColorEncoder&) = delete;
        ColorEncoder& operator=(ColorEncoder&&) = delete;

        virtual float FromLinear(const float v) const = 0;
        virtual float ToLinear(const float v) const = 0;
    };

    class LinearEncoder : public ColorEncoder {
    public:
        LinearEncoder() = default;
        virtual ~LinearEncoder() {}

        float FromLinear(const float v) const override
        {
            return v;
        }

        float ToLinear(const float v) const override
        {
            return v;
        }
    };

    class sRGBGammaEncoder : public ColorEncoder {
    public:
        sRGBGammaEncoder() = default;
        virtual ~sRGBGammaEncoder() {}

        float FromLinear(const float v) const override
        {
            if (v <= 0.0031308F) {
                return v * 12.92F;
            }
            return 1.055F * aten::pow(v, 1 / 2.4F) - 0.055F;
        }

        float ToLinear(const float v) const override
        {
            if (v <= 0.04045F) {
                return v / 12.92F;
            }
            return aten::pow((v + 0.055F) / 1.055F, 2.4F);
        }
    };
}
