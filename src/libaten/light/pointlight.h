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

        /**
         * @brief Constructor.
         * @param[in] pos Light position.
         * @param[in] light_color Light color.
         * @param[in] intensity Punctual light intensity [W] as point light.
         * @param[in] scale Scale for the sampled light color.
         */
        PointLight(
            const aten::vec3& pos,
            const aten::vec3& light_color,
            const float intensity,
            const float scale = 1.0F)
            : Light(aten::LightType::Point, aten::LightAttributeSingluar)
        {
            param_.pos = pos;
            param_.light_color = light_color;
            param_.intensity = intensity;
            param_.scale = scale;
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

            const auto dist2 = aten::sqr(result.dist_to_light);

            // Convert intensity [W] to [W/m^2] by dividing with the squared distance.
            result.light_color = param.light_color * param.scale * param.intensity / dist2;
        }
    };
}
