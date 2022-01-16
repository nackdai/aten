#pragma once

#include "kernel/idatendefs.cuh"
#include "kernel/pt_params.h"
#include "kernel/context.cuh"
#include "kernel/accelerator.cuh"

namespace kernel {
    inline __device__ bool hitShadowRay(
        bool enableLod,
        const idaten::Context& ctxt,
        const idaten::ShadowRay& shadowRay)
    {
        auto targetLightId = shadowRay.targetLightId;
        auto distToLight = shadowRay.distToLight;

        auto light = ctxt.lights[targetLightId];
        auto lightobj = (light.objid >= 0 ? &ctxt.shapes[light.objid] : nullptr);

        real distHitObjToRayOrg = AT_MATH_INF;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::GeomParameter* hitobj = lightobj;

        aten::Intersection isectTmp;

        bool isHit = false;

        aten::ray r(shadowRay.rayorg, shadowRay.raydir);

        isHit = intersectCloser(&ctxt, r, &isectTmp, distToLight - AT_MATH_EPSILON, enableLod);

        if (isHit) {
            hitobj = &ctxt.shapes[isectTmp.objid];
        }

        isHit = AT_NAME::scene::hitLight(
            isHit,
            light.attrib,
            lightobj,
            distToLight,
            distHitObjToRayOrg,
            isectTmp.t,
            hitobj);

        return isHit;
    }

    inline __device__ void hitImplicitLight(
        bool is_back_facing,
        int bounce,
        idaten::PathContrib& path_contrib,
        idaten::PathAttribute& path_attrib,
        idaten::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        float hit_area,
        const aten::MaterialParameter& mtrl)
    {
        if (!is_back_facing) {
            float weight = 1.0f;

            if (bounce > 0 && !path_attrib.isSingular) {
                auto cosLight = dot(hit_nml, -ray.dir);
                auto dist2 = aten::squared_length(hit_pos - ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / hit_area;

                    // Convert pdf area to sradian.
                    // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    weight = path_throughput.pdfb / (pdfLight + path_throughput.pdfb);
                }
            }

            auto contrib = path_throughput.throughput * weight * static_cast<aten::vec3>(mtrl.baseColor);
            path_contrib.contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }

        // When ray hit the light, tracing will finish.
        path_attrib.isTerminate = true;
    }
}
