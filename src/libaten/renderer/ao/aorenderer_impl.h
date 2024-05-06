#pragma once

#include "geometry/EvaluateHitResult.h"
#include "material/material_impl.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "sampler/cmj.h"
#include "scene/scene.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/accelerator.cuh"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME {
namespace ao {
    /**
     * @brief Shade Ambient Occlusion.
     *
     * @param[in] idx Inded to the shading pixel.
     * @param[in] frame Frame count.
     * @param[in] rnd Seed for random.
     * @param[in] ao_num_rays Number of ray to shade AO.
     * @param[in] ao_radius Radius of the sphere to shader AO.
     * @param[in,out] paths Information of paths.
     * @param[in] ctxt Scene context.
     * @param[in] ray Query ray.
     * @param[in] isect Scene intersection information.
     * @param[in,out] scene Scene instance. Only for running on host.
     */
    template <class SCENE = void>
    inline AT_DEVICE_API void ShandeAO(
        int32_t idx,
        uint32_t frame, uint32_t rnd,
        int32_t ao_num_rays, float ao_radius,
        AT_NAME::Path& paths,
        const context& ctxt,
        const aten::ray& ray,
        const aten::Intersection& isect,
        SCENE* scene = nullptr)
    {
        auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 5 * 300, scramble);

        aten::hitrecord rec;

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        aten::vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        // Apply normal map.
        AT_NAME::material::applyNormal(
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, &paths.sampler[idx]);

        aten::vec3 ao_color{ 0.0f };

        for (int32_t i = 0; i < ao_num_rays; i++) {
            auto nextDir = AT_NAME::lambert::sampleDirection(orienting_normal, &paths.sampler[idx]);
            auto pdfb = AT_NAME::lambert::pdf(orienting_normal, nextDir);

            float c = dot(orienting_normal, nextDir);

            auto ao_ray = aten::ray(rec.p, nextDir, orienting_normal);

            aten::Intersection ao_isect;

            bool isHit = false;

            if constexpr (!std::is_void_v<std::remove_pointer_t<SCENE>>) {
                if (scene) {
                    isHit = scene->hit(ctxt, ao_ray, AT_MATH_EPSILON, ao_radius, ao_isect);
                }
            }
            else {
#ifndef __CUDACC__
                // Dummy to build with clang.
                auto intersectClosest = [](auto... args) -> bool { return true; };
#endif
                isHit = intersectClosest(&ctxt, ao_ray, &ao_isect, ao_radius);
            }

            if (isHit) {
                if (c > 0.0f) {
                    ao_color += aten::vec3(ao_isect.t / ao_radius * c / pdfb);
                }
            }
            else {
                ao_color = aten::vec3(1.0f);
            }
        }

        ao_color /= ao_num_rays;
        _detail::CopyVec(paths.contrib[idx].contrib, ao_color);
    }

    /**
     * @breif Shade Ambient Occulusion, if hit test is missed.
     *
     * @param[in] idx Index to the shading pixel.
     * @param[in] is_first_bounce Specify if this function is called for the first bounce.
     * @param[in,out] paths Information of paths.
     */
    inline AT_DEVICE_API void ShadeMissAO(
        int32_t idx,
        bool is_first_bounce,
        AT_NAME::Path& paths)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            if (is_first_bounce) {
                paths.attrib[idx].isKill = true;
            }

            _detail::CopyVec(paths.contrib[idx].contrib, aten::vec3(1));

            paths.attrib[idx].isTerminate = true;
        }
    }
}
}
