#pragma once

#include "material/toon.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/accelerator.cuh"
#else
#include "scene/host_scene_context.h"
#endif

#include "accelerator/threaded_bvh_traverser.h"

namespace AT_NAME
{
    // NOTE:
    // If we implement the following code in toon.h, toon.h is included in material.cpp.
    // But, it causes the compile error not to find intersectCloser at compiling cuda code.
    // To avoid it, split the following code in this file not to be included in material.cpp.

    template <class SCENE/*= void*/>
    inline AT_DEVICE_API aten::vec3 Toon::bsdf(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& param,
        aten::sampler& sampler,
        const aten::vec3& hit_pos,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        SCENE* scene/*= nullptr*/)
    {
        // Pick target light.
        const auto* target_light = param.toon.target_light_idx >= 0
            ? &ctxt.GetLight(param.toon.target_light_idx)
            : nullptr;

#if 0
        // Allow only singular light.
        target_light = target_light && target_light->attrib.is_singular
            ? target_light
            : nullptr;
#endif

        aten::vec3 brdf{ 0.0F };

        if (target_light) {
            // NOTE:
            // The target light has to be the singular light.
            // In that case, sample is not used at all. So, we can pass it as nullptr.
            aten::LightSampleResult light_sample;
            AT_NAME::Light::sample(light_sample, *target_light, ctxt, hit_pos, normal, &sampler);

            aten::ray r(hit_pos, light_sample.dir, normal);

            aten::Intersection isect;

            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closer>(
                isect,
                ctxt,
                r,
                AT_MATH_EPSILON, light_sample.dist_to_light - AT_MATH_EPSILON);

            if (param.type == aten::MaterialType::Toon) {
                brdf = Toon::ComputeBRDF(
                    ctxt, param,
                    is_hit ? nullptr : &light_sample,
                    sampler, hit_pos, normal, wi, u, v);
            }
            else if (param.type == aten::MaterialType::StylizedBrdf) {
                brdf = StylizedBrdf::ComputeBRDF(
                    ctxt, param,
                    is_hit ? nullptr : &light_sample,
                    sampler, hit_pos, normal, wi, u, v);
            }
        }

        return brdf;
    }
}
