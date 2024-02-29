#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class SpotLight : public Light {
    public:
        SpotLight()
            : Light(aten::LightType::Spot, aten::LightAttributeSingluar)
        {}
        SpotLight(
            const aten::vec3& pos,  // light position.
            const aten::vec3& dir,  // light direction from the position.
            const aten::vec3& light_color,   // light color.
            float flux,
            float innerAngle,    // Umbra angle of spotlight in radians.
            float outerAngle)    // Penumbra angle of spotlight in radians.
            : Light(aten::LightType::Spot, aten::LightAttributeSingluar)
        {
            m_param.pos = pos;
            m_param.dir = normalize(dir);
            m_param.light_color = light_color;

            // Convert flux[W] to intensity[W/sr]
            m_param.intensity = flux / AT_MATH_PI;

            setSpotlightFactor(innerAngle, outerAngle);
        }

        SpotLight(aten::Values& val);

        virtual ~SpotLight() = default;

    public:
        void setSpotlightFactor(
            float innerAngle,    // Umbra angle of spotlight in radians.
            float outerAngle)    // Penumbra angle of spotlight in radians.
        {
            m_param.innerAngle = aten::clamp<float>(innerAngle, 0, AT_MATH_PI - AT_MATH_EPSILON);
            m_param.outerAngle = aten::clamp<float>(outerAngle, innerAngle, AT_MATH_PI - AT_MATH_EPSILON);
        }

        static AT_HOST_DEVICE_API void sample(
            const aten::LightParameter& param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult& result)
        {
            result.pdf = 1.0f;
            result.pos = param.pos;
            result.dir = ((aten::vec3)param.pos) - org;
            result.nml = param.dir;   // already normalized

            auto dir_to_light = -normalize(result.dir);

            auto rho = dot((aten::vec3)param.dir, dir_to_light);

            auto cosHalfInner = aten::cos(param.innerAngle * float(0.5));
            auto cosHalfOuter = aten::cos(param.outerAngle * float(0.5));

            if (rho > cosHalfOuter) {
                auto angle_attenuation = (rho - cosHalfOuter) / (cosHalfInner - cosHalfOuter);
                angle_attenuation = aten::clamp<float>(angle_attenuation, 0.0f, 1.0f);

                auto dist2 = aten::squared_length(result.dir);

                auto luminance = param.scale * param.intensity / dist2 / AT_MATH_PI;
                result.light_color = param.light_color * luminance;
            }
            else {
                // Out of spot light.
                result.pdf = 0.0f;
                aten::set(result.light_color, 0, 0, 0);
            }
        }
    };
}
