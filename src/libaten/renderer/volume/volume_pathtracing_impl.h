#pragma once

#include "defs.h"
#include "material/material.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "misc/tuple.h"
#include "renderer/pathtracing/pt_params.h"
#include "volume/medium.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif


namespace AT_NAME
{
    inline AT_DEVICE_API void UpdateMedium(
        const ray& ray,
        const vec3& wo,
        const vec3& surface_normal,
        const MaterialParameter& mtrl,
        aten::MedisumStack& mediums)
    {
        const auto wi = -ray.dir;
        const auto is_trasmitted = dot(wo, surface_normal) < 0 != dot(wi, surface_normal) < 0;
        const auto is_enter = dot(wi, surface_normal) > 0;

        if (is_trasmitted) {
            if (is_enter) {
                if (mtrl.is_medium) {
                    mediums.push(mtrl.id);
                }
            }
            else {
                //std::ignore = throughput.mediums.safe_pop();
                if (mediums.size() > 0) {
                    mediums.pop();
                }
            }
        }
    }

    inline AT_DEVICE_API const aten::MediumParameter& GetCurrentMedium(
        const AT_NAME::context& ctxt,
        const aten::MedisumStack& mediums)
    {
        auto idx = mediums.top();
        return ctxt.GetMaterial(idx).medium;
    }

    inline AT_DEVICE_API bool HasMedium(const aten::MedisumStack& mediums)
    {
        return !mediums.empty();
    }

    inline AT_DEVICE_API bool IsSubsurface(const aten::MaterialParameter& mtrl)
    {
        return mtrl.type != aten::MaterialType::MaterialTypeMax && mtrl.is_medium;
    }

    inline AT_DEVICE_API aten::tuple<bool, float> TraverseShadowRay(
        const AT_NAME::context& ctxt,
        const aten::LightSampleResult& light_sample,
        const aten::vec3& start_point,
        const aten::vec3& surface_nml,
        aten::MedisumStack medium_stack,
        scene* scene)
    {
        float transmittance = 1.0F;

        auto nml = dot(light_sample.dir, surface_nml) > 0
            ? surface_nml
            : -surface_nml;
        aten::ray ray(start_point, light_sample.dir, nml);

        // Need to check if hitting to something closer than light.
        // So, set shorter distance to light.
        auto t_max = light_sample.dist_to_light - AT_MATH_EPSILON;

        while (true) {
            aten::Intersection isect;
            bool is_hit = scene->hit(ctxt, ray, AT_MATH_EPSILON, t_max, isect, aten::HitStopType::Closer);

            if (is_hit) {
                const auto& hit_obj = ctxt.GetObject(isect.objid);
                aten::hitrecord hrec;
                aten::evaluate_hit_result(hrec, hit_obj, ctxt, ray, isect);

                const auto& mtrl = ctxt.GetMaterial(hrec.mtrlid);

                // NOTE:
                // In the case that meidum is filled within the surface (i.e. sub surface),
                // if the ray goes out from the surface, it can be treated as it's still in the medium and continue to traverse.
                // To check whether it's the above case, we need to check whether the ray enters the surface.
                bool is_enter = dot(-ray.dir, hrec.normal) > 0;

                if (!mtrl.is_medium || is_enter) {
                    // Hit surface to occlude light.
                    return aten::make_tuple(false, transmittance);
                }

                // Hit medium. So, deal with it as not to occlude light.

                if (HasMedium(medium_stack)) {
                    const auto& medium = GetCurrentMedium(ctxt, medium_stack);
                    const auto tr = AT_NAME::HomogeniousMedium::TransmittanceFromMediumParam(
                        medium,
                        ray.org, hrec.p);
                    transmittance *= tr;
                }

                UpdateMedium(ray, ray.dir, hrec.normal, mtrl, medium_stack);

                // Advance ray.
                nml = dot(ray.dir, hrec.normal) > 0
                    ? hrec.normal
                    : -hrec.normal;
                ray = aten::ray(hrec.p, ray.dir, nml);
                t_max -= isect.t;
            }
            else {
                // Nothing to occlude to light.
                if (HasMedium(medium_stack)) {
                    const auto& medium = GetCurrentMedium(ctxt, medium_stack);
                    const auto tr = AT_NAME::HomogeniousMedium::TransmittanceFromMediumParam(
                        medium, t_max);
                    transmittance *= tr;
                }

                // No need to traverse any more.
                break;
            }
        }

        return aten::make_tuple(true, transmittance);
    }
}
