#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class DirectionalLight : public Light {
    public:
        DirectionalLight()
            : Light(aten::LightType::Direction, aten::LightAttributeDirectional)
        {}
        DirectionalLight(
            const aten::vec3& dir,
            const aten::vec3& light_color,
            float luminance)
            : Light(aten::LightType::Direction, aten::LightAttributeDirectional)
        {
            m_param.dir = normalize(dir);
            m_param.light_color = light_color;
            m_param.luminance = luminance;
        }

        DirectionalLight(aten::Values& val);

        virtual ~DirectionalLight() = default;

    public:
        static AT_HOST_DEVICE_API void sample(
            const aten::LightParameter& param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult& result)
        {
            result.pdf = 1.0f;

            result.dir = -normalize(param.dir);
            result.nml = normalize(param.dir);

            // TODO
            // Light should be located at sphere to cover entire scene.
            result.pos = org + float(100000) * float(0.5) * result.dir;

            result.light_color = param.light_color * param.scale * param.luminance;

            // NOTE:
            // Theoretically this should be inf.
            // But, to compute the geometry term by divinding the squared distance without checking the light type,
            // the distance to light is 1.0 is helpful.
            result.dist_to_light = 1.0F;
        }
    };
}
