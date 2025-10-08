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

        /**
         * @brief Constructor.
         * @param[in] pos Light position.
         * @param[in] dir Light direction from the position.
         * @param[in] light_color Light color.
         * @param[in] intensity Punctual light intensity [W] as spot light.
         * @param[in] inner_angle Umbra angle of spotlight in radians.
         * @param[in] outer_angle Penumbra angle of spotlight in radians.
         * @param[in] scale Scale for the sampled light color.
         */
        SpotLight(
            const aten::vec3& pos,
            const aten::vec3& dir,
            const aten::vec3& light_color,
            const float intensity,
            const float inner_angle,
            const float outer_angle,
            const float scale = 1.0F)
            : Light(aten::LightType::Spot, aten::LightAttributeSingluar)
        {
            param_.pos = pos;
            param_.dir = normalize(dir);
            param_.light_color = light_color;
            param_.intensity = intensity;
            param_.scale = scale;

            setSpotlightFactor(inner_angle, outer_angle);
        }

        SpotLight(aten::Values& val);

        virtual ~SpotLight() = default;

    public:
        void setSpotlightFactor(
            float inner_angle,    // Umbra angle of spotlight in radians.
            float outer_angle)    // Penumbra angle of spotlight in radians.
        {
            param_.innerAngle = aten::clamp<float>(inner_angle, 0, AT_MATH_PI - AT_MATH_EPSILON);
            param_.outerAngle = aten::clamp<float>(outer_angle, inner_angle, AT_MATH_PI - AT_MATH_EPSILON);
        }

        static AT_HOST_DEVICE_API void sample(
            const aten::LightParameter& param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult& result)
        {
            result.pdf = 1.0f;
            result.pos = param.pos;
            result.nml = param.dir;   // already normalized

            result.dir = ((aten::vec3)param.pos) - org;
            result.dist_to_light = length(result.dir);
            result.dir = normalize(result.dir);

            auto dir_to_light = -result.dir;

            auto rho = dot((aten::vec3)param.dir, dir_to_light);

            auto cosHalfInner = aten::cos(param.innerAngle * float(0.5));
            auto cosHalfOuter = aten::cos(param.outerAngle * float(0.5));

            if (rho > cosHalfOuter) {
                auto angle_attenuation = (rho - cosHalfOuter) / (cosHalfInner - cosHalfOuter);
                angle_attenuation = aten::clamp<float>(angle_attenuation, 0.0f, 1.0f);

                // Convert intensity [W] to [W/m^2] by dividing with the squared distance.
                auto dist2 = aten::sqr(result.dist_to_light);
                result.light_color = param.scale * param.light_color * angle_attenuation * param.intensity / dist2;
            }
            else {
                // Out of spot light.
                result.pdf = 0.0f;
                aten::set(result.light_color, 0, 0, 0);
            }
        }
    };
}
