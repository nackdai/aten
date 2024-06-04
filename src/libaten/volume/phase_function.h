#pragma once

#include "defs.h"
#include "math/vec3.h"

namespace AT_NAME {
    class HenyeyGreensteinPhaseFunction {
    private:
        HenyeyGreensteinPhaseFunction() = delete;
        ~HenyeyGreensteinPhaseFunction() = delete;

    public:
        static inline AT_DEVICE_API float Evaludate(
            float g,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            g = aten::clamp(g, -1.0F, 1.0F);
            const auto g2 = aten::sqr(g);
            const auto costheta = dot(wi, wo);
            const auto _4pi = 4 * AT_MATH_PI;

            const float f = (1 - g2) / (_4pi * aten::pow(1 + g2 - 2 * g * costheta, 1.5F));
            return f;
        }

        static inline AT_DEVICE_API float SamplePDF(
            float g,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            return Evaludate(g, wi, wo);
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            float g,
            const aten::vec3& w)
        {
            g = aten::clamp(g, -1.0F, 1.0F);

            auto costheta = 0.0F;
            if (aten::abs(g) < AT_MATH_EPSILON) {
                costheta = 1 - 2 * r1;
            }
            else {
                const auto g2 = aten::sqr(g);
                costheta = 1 / (2 * g) * (1 + g2 - aten::sqr((1 - g2) / (1 - g + 2 * g * r1)));
            }

            const auto phi = AT_MATH_PI_2 * r2;
            const auto sintheta = aten::sqrt(1 - costheta * costheta);

            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sqrt(1 - cosphi * cosphi);

            auto t = aten::getOrthoVector(w);
            auto b = cross(w, t);

            auto dir = t * sintheta * cosphi + b * sintheta * sinphi + w * costheta;
            dir = normalize(dir);

            return dir;
        }
    };
}
