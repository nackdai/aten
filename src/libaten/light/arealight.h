#pragma once

#include <memory>

#include "geometry/EvaluateHitResult.h"
#include "light/light.h"

namespace aten {
    class Values;
    class transformable;
}

namespace AT_NAME {
    class AreaLight : public Light {
    public:
        AreaLight()
            : Light(aten::LightType::Area, aten::LightAttributeArea)
        {}

        /**
         * @brief Constructor.
         * @param[in] pos Light position.
         * @param[in] light_color Light color.
         * @param[in] intensity Light intensity as radiant flux [W].
         * @param[in] scale Scale for the sampled light color.
         */
        AreaLight(
            const std::shared_ptr<aten::transformable>& obj,
            const aten::vec3& light_color,
            const float intensity,
            const float scale = 1.0F);

        AreaLight(const aten::Values& val);

        virtual ~AreaLight() = default;

    public:
        static AT_HOST_DEVICE_API void sample(
            const aten::hitrecord& hrec,
            const aten::LightParameter& param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult& result)
        {
            result.pos = hrec.p;

            result.pdf = 1 / hrec.area;

            result.dir = hrec.p - org;
            result.dist_to_light = length(result.dir);
            result.dir = normalize(result.dir);

            result.nml = hrec.normal;

            result.light_color = ComputeLightColor(param, hrec.area);
        }

        static AT_HOST_DEVICE_API aten::vec3 ComputeLightColor(const aten::LightParameter& param, float area)
        {
            // Convert radiant flux[W] to irradiance[W/m^2]
            auto luminance = param.scale * param.intensity / area;
            return param.light_color* luminance;
        }

        template <class CONTEXT>
        static AT_HOST_DEVICE_API void sample(
            aten::LightSampleResult& result,
            const aten::LightParameter& param,
            const CONTEXT& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler)
        {
            if (param.arealight_objid < 0) {
                return;
            }

            bool isHit = false;

            const auto& obj = ctxt.GetObject(param.arealight_objid);

            aten::ray r;
            aten::hitrecord rec;
            aten::Intersection isect;

            if (sampler) {
                aten::SamplePosNormalPdfResult result;

                AT_NAME::SamplePosAndNormal(&result, obj, ctxt, sampler);

                auto pos = result.pos;
                auto dir = pos - org;

                // NOTE:
                // If ray hits the specified object directly, we will just do hit test.
                // We don't need to mind self-intersection.
                // Therefore, we don't need to add offset.
                // i.e. We don't use normal vector with adding offset to ray.
                r = aten::ray(org, dir);

                if (result.triangle_id >= 0) {
                    isect.t = length(dir);

                    isect.hit.tri.id = result.triangle_id;

                    isect.hit.tri.a = result.a;
                    isect.hit.tri.b = result.b;

                    // We can treat as hit.
                    isHit = true;
                }
                else {
                    // No triangle means object shape is geometric form.
                    // Currently, support only sphere.
                    isHit = sphere::hit(&obj, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
                }
            }
            else {
                // In no sampler case, we can suport only geometric form.
                // Currently, support only sphere.
                auto pos = obj.sphere.center;
                auto dir = pos - org;

                // NOTE:
                // If ray hits the specified object directly, we will just do hit test.
                // We don't need to mind self-intersection.
                // Therefore, we don't need to add offset.
                // i.e. We don't use normal vector with adding offset to ray.
                r = aten::ray(org, dir);

                isHit = sphere::hit(&obj, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
            }

            if (isHit) {
                AT_NAME::evaluate_hit_result(rec, obj, ctxt, r, isect);

                sample(
                    rec,
                    param,
                    org,
                    sampler,
                    result);
            }
        }

        std::shared_ptr<aten::transformable> getLightObject() const
        {
            return m_obj;
        }

    private:
        std::shared_ptr<aten::transformable> m_obj;
    };
}
