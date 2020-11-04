#include "light/arealight.h"

namespace AT_NAME {
    aten::LightSampleResult AreaLight::sample(
        const aten::context& ctxt,
        const aten::vec3& org,
        aten::sampler* sampler) const
    {
        bool isHit = false;
        auto obj = getLightObject();

        aten::LightSampleResult result;

        if (obj) {
            aten::ray r;
            aten::hitrecord rec;
            aten::Intersection isect;

            if (sampler) {
                aten::hitable::SamplePosNormalPdfResult result;

                obj->getSamplePosNormalArea(ctxt, &result, sampler);

                auto pos = result.pos;
                auto dir = pos - org;

                // NOTE
                // Just do hit test if ray hits the specified object directly.
                // We don't need to mind self-intersection.
                // Therefore, we don't need to add offset.
                r = aten::ray(org, dir);

                if (result.primid >= 0) {
                    isect.t = length(dir);

                    isect.primid = result.primid;

                    isect.a = result.a;
                    isect.b = result.b;
                }
                else {
                    obj->hit(ctxt, r, AT_MATH_EPSILON, AT_MATH_INF, isect);
                }
            }
            else {
                auto pos = obj->getBoundingbox().getCenter();

                auto dir = pos - org;

                // NOTE
                // Just do hit test if ray hits the specified object directly.
                // We don't need to mind self-intersection.
                // Therefore, we don't need to add offset.
                r = aten::ray(org, dir);

                obj->hit(ctxt, r, AT_MATH_EPSILON, AT_MATH_INF, isect);
            }

            aten::hitable::evalHitResultForAreaLight(ctxt, obj, r, rec, isect);

            sample(
                &rec,
                &this->param(),
                org,
                sampler,
                &result);

            result.obj = m_obj;
        }

        return std::move(result);
    }
}
