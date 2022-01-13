#pragma once

#include "kernel/idatendefs.cuh"
#include "kernel/pt_params.h"
#include "kernel/context.cuh"
#include "kernel/accelerator.cuh"

namespace kernel {
    AT_CUDA_INLINE __device__ bool hitShadowRay(
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
}
