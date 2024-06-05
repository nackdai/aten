#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class PointLight : public Light {
    public:
        PointLight()
            : Light(aten::LightType::Point, aten::LightAttributeSingluar)
        {}
        PointLight(
            const aten::vec3& pos,
            const aten::vec3& light_color,
            float flux)
            : Light(aten::LightType::Point, aten::LightAttributeSingluar)
        {
            m_param.pos = pos;
            m_param.light_color = light_color;

            // Convert flux[W] to intensity[W/sr]
            m_param.intensity = flux / (4.0f * AT_MATH_PI);
        }

        PointLight(aten::Values& val);

        virtual ~PointLight() = default;

    public:
        static AT_HOST_DEVICE_API void sample(
            const aten::LightParameter& param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult& result)
        {
            result.pdf = 1.0f;

            result.dir = ((aten::vec3)param.pos) - org;
            result.dist_to_light = length(result.dir);
            result.dir = normalize(result.dir);

            result.pos = param.pos;
            result.nml = normalize(-result.dir);

            auto dist2 = aten::sqr(result.dist_to_light);

            auto luminance = param.scale * param.intensity / dist2;
            result.light_color = param.light_color * luminance;
        }
    };
}
