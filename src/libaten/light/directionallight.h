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

        /**
         * @brief Constructor.
         * @param[in] pos Light position.
         * @param[in] light_color Light color.
         * @param[in] intensity Punctual light intensity [W/m^2] as directional light.
         * @param[in] scale Scale for the sampled light color.
         */
        DirectionalLight(
            const aten::vec3& dir,
            const aten::vec3& light_color,
            const float intensity,
            const float scale = 1.0F)
            : Light(aten::LightType::Direction, aten::LightAttributeDirectional)
        {
            m_param.dir = normalize(dir);
            m_param.light_color = light_color;
            m_param.intensity = intensity;
            m_param.scale = scale;
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

            result.light_color = param.light_color * param.scale * param.intensity;

            // NOTE:
            // Theoretically this should be inf.
            // But, to compute the geometry term by divinding the squared distance without checking the light type,
            // the distance to light is 1.0 is helpful.
            result.dist_to_light = 1.0F;
        }
    };
}
