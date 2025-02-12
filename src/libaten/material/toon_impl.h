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
        float& pdf,
        SCENE* scene/*= nullptr*/)
    {
        // Pick target light.
        const auto* target_light = param.toon.target_light_idx >= 0
            ? &ctxt.GetLight(param.toon.target_light_idx)
            : nullptr;

        // Allow only singular light.
        target_light = target_light && target_light->attrib.is_singular
            ? target_light
            : nullptr;

        aten::vec3 brdf{ 0.0F };

        if (target_light) {
            // NOTE:
            // The target light has to be the singular light.
            // In that case, sample is not used at all. So, we can pass it as nullptr.
            aten::LightSampleResult light_sample;
            AT_NAME::Light::sample(light_sample, *target_light, ctxt, hit_pos, normal, &sampler);

            bool is_hit = false;

            aten::ray r(hit_pos, light_sample.dir, normal);

            aten::Intersection isect;

            if constexpr (!std::is_void_v<std::remove_pointer_t<SCENE>>) {
                // NOTE:
                // operation has to be related with template arg SCENE.
                if (scene) {
                    is_hit = scene->hit(ctxt, r, AT_MATH_EPSILON, light_sample.dist_to_light - AT_MATH_EPSILON, isect);
                }
            }
            else {
#ifndef __CUDACC__
                // Dummy to build with clang.
                auto intersectCloser = [](auto... args) -> bool { return true; };
#endif
                is_hit = intersectCloser(&ctxt, r, &isect, light_sample.dist_to_light - AT_MATH_EPSILON, 0);
            }

            brdf = ComputeBRDF(
                ctxt, param,
                is_hit ? nullptr : &light_sample,
                sampler, hit_pos, normal, wi, u, v, pdf);
        }

        return brdf;
    }
}
