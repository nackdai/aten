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
        AreaLight(
            const std::shared_ptr<aten::transformable>& obj,
            const aten::vec3& light_color,
            float lux);

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

            // Convert intenstiy[W/sr] to luminance[W/(sr * m^2)]
            auto luminance = param.scale * param.intensity / hrec.area;
            result.light_color = param.light_color * luminance;
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

                    isect.triangle_id = result.triangle_id;

                    isect.a = result.a;
                    isect.b = result.b;

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
