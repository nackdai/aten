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
            float ret = float(0.2126) * r + float(0.7152) * g + float(0.0722) * b;
            return ret;
        }

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
