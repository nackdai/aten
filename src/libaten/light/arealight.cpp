#include "light/arealight.h"
#include "geometry/EvaluateHitResult.h"

namespace AT_NAME {
    aten::LightSampleResult AreaLight::sample(
        const aten::context& ctxt,
        const aten::vec3& org,
        aten::sampler* sampler) const
    {
        bool isHit = false;
        const auto& obj = getLightObject();

        aten::LightSampleResult result;

        if (obj) {
            aten::ray r;
            aten::hitrecord rec;
            aten::Intersection isect;

            if (sampler) {
                aten::SamplePosNormalPdfResult result;

                AT_NAME::sample_pos_and_normal(&result, obj->getParam(), ctxt, sampler);

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
                    isHit = obj->hit(ctxt, r, AT_MATH_EPSILON, AT_MATH_INF, isect);
                }
            }
            else {
                auto pos = obj->getBoundingbox().getCenter();

                auto dir = pos - org;

                // NOTE:
                // If ray hits the specified object directly, we will just do hit test.
                // We don't need to mind self-intersection.
                // Therefore, we don't need to add offset.
                // i.e. We don't use normal vector with adding offset to ray.
                r = aten::ray(org, dir);

                isHit = obj->hit(ctxt, r, AT_MATH_EPSILON, AT_MATH_INF, isect);
            }

            if (isHit) {
                AT_NAME::evaluate_hit_result(rec, obj->getParam(), ctxt, r, isect);

                sample(
                    &rec,
                    &this->param(),
                    org,
                    sampler,
                    &result);

                result.obj = m_obj.get();
            }
        }

        return result;
    }

    void AreaLight::getSamplePosNormalArea(
        const aten::context& ctxt,
        aten::SamplePosNormalPdfResult* result,
        aten::sampler* sampler) const
    {
        if (m_obj) {
            auto obj = getLightObject();
            AT_NAME::sample_pos_and_normal(result, obj->getParam(), ctxt, sampler);
        }
    }
}
