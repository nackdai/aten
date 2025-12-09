#pragma once

#include "accelerator/threaded_bvh_traverser.h"
#include "geometry/EvaluateHitResult.h"
#include "material/material_impl.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "sampler/cmj.h"
#include "scene/scene.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME {
namespace ao {
    /**
     * @brief Shade Ambient Occlusion.
     *
     * @param[in] ao_num_rays Number of ray to shade AO.
     * @param[in] ao_radius Radius of the sphere to shader AO.
     * @param[in,out] sampler Sampler.
     * @param[in] ctxt Scene context.
     * @param[in] ray Query ray.
     * @param[in] isect Scene intersection information.
     * @param[in,out] scene Scene instance. Only for running on host.
     * @return Shaded color by ambient occlusion.
     */
    inline AT_DEVICE_API float ShandeByAO(
        int32_t ao_num_rays, float ao_radius,
        aten::sampler& sampler,
        const context& ctxt,
        const aten::ray& ray,
        const aten::Intersection& isect
    )
    {
        aten::hitrecord rec;

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        aten::vec3 orienting_normal = rec.normal;

        // To get normal map.
        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        // Apply normal map.
        AT_NAME::material::applyNormal(
            ctxt,
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, &sampler);

        float ao_color{ 0.0F };

        for (int32_t i = 0; i < ao_num_rays; i++) {
            auto nextDir = AT_NAME::Diffuse::sampleDirection(orienting_normal, &sampler);
            auto pdfb = AT_NAME::Diffuse::pdf(orienting_normal, nextDir);

            float c = dot(orienting_normal, nextDir);

            auto ao_ray = aten::ray(rec.p, nextDir, orienting_normal);

            aten::Intersection ao_isect;

            bool isHit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                ao_isect,
                ctxt,
                ao_ray,
                AT_MATH_EPSILON, ao_radius);

            if (isHit) {
                if (c > 0.0f) {
                    ao_color += ao_isect.t / ao_radius * c / pdfb;
                }
            }
            else {
                ao_color = 1.0f;
            }
        }

        ao_color /= ao_num_rays;
        return ao_color;
    }

    /**
     * @breif Shade Ambient Occulusion, if hit test is missed.
     *
     * @param[in] idx Index to the shading pixel.
     * @param[in,out] paths Information of paths.
     * @return Check if ray hits to something, and if ray hits to something, return 1.0F.
     *         Otherwize, return netative value as invalid value;
     */
    inline AT_DEVICE_API float ShadeByAOIfHitMiss(
        int32_t idx,
        AT_NAME::Path& paths)
    {
        if (!paths.attrib[idx].attr.is_terminated && !paths.attrib[idx].attr.isHit) {
            paths.attrib[idx].attr.is_terminated = true;
            return 1.0F;
        }
        return -1.0F;
    }

    template <class SRC, class DST, bool IsHorizontal, int32_t KernelSize = 3>
    inline AT_DEVICE_API DST ApplyBilateralFilter(
        const int32_t center_x, const int32_t center_y,
        const int32_t width, const int32_t height,
        const float coeff_pixel_dist,
        const float coeff_depth,
        const SRC* values,
        const aten::Intersection* isects
    )
    {
        const auto coeff_pixel_dist_2 = 2 * (coeff_pixel_dist * coeff_pixel_dist);
        const auto coeff_depth_2 = 2 * (coeff_depth * coeff_depth);

        const auto center_idx = center_y * width + center_x;
        const auto center_depth = isects[center_idx].t;

        DST numer = 0.0F;
        DST denom = 0.0F;
        int32_t count = 0;

#pragma unroll
        for (int32_t i = -KernelSize; i <= KernelSize; i++) {
            auto x = center_x;
            auto y = center_y;
            if constexpr (IsHorizontal) {
                x = aten::clamp(center_x + i, 0, width - 1);
            }
            else {
                y = aten::clamp(center_y + i, 0, height - 1);
            }
            const auto idx = y * width + x;

            const auto depth = isects[idx].t;
            const auto diff = center_depth - depth;
            const auto kernel = aten::exp(-(i * i) / coeff_pixel_dist_2 - (diff * diff) / center_depth);

            DST value;
            if constexpr (std::is_same_v<SRC, AT_NAME::PathContrib>) {
                value = values[idx].contrib.x;
            }
            else {
                value = values[idx];
            }

            numer += value * kernel;
            denom += kernel;
        }

        const auto result = aten::clamp(denom > 0.0F ? numer / denom : 1.0F, 0.0F, 1.0F);
        return result;
    }

    template <class SRC, class DST, int32_t KernelSizeH = 3, int32_t KernelSizeV = 3>
    inline AT_DEVICE_API DST ApplyBilateralFilterOrthogonal(
        const int32_t center_x, const int32_t center_y,
        const int32_t width, const int32_t height,
        const float coeff_pixel_dist,
        const float coeff_depth,
        const SRC* values,
        const aten::Intersection* isects
    )
    {
        const auto coeff_pixel_dist_2 = 2 * (coeff_pixel_dist * coeff_pixel_dist);
        const auto coeff_depth_2 = 2 * (coeff_depth * coeff_depth);

        const auto center_idx = center_y * width + center_x;
        const auto center_depth = isects[center_idx].t;

        DST numer = 0.0F;
        DST denom = 0.0F;
        int32_t count = 0;

        constexpr auto inclination = static_cast<float>(KernelSizeH) / KernelSizeV;

#pragma unroll
        for (int32_t h = -KernelSizeH; h <= KernelSizeH; h++) {
            const auto step_v = static_cast<int32_t>(inclination * h);
            const auto x = aten::clamp(center_x + step_v, 0, width - 1);
            const auto y = aten::clamp(center_y + h, 0, height - 1);
            const auto i = aten::sqrt(step_v * step_v + h * h);
            const auto idx = y * width + x;

            const auto depth = isects[idx].t;
            const auto diff = center_depth - depth;
            const auto kernel = aten::exp(-(i * i) / coeff_pixel_dist_2 - (diff * diff) / center_depth);

            DST value;
            if constexpr (std::is_same_v<SRC, AT_NAME::PathContrib>) {
                value = values[idx].contrib.x;
            }
            else {
                value = values[idx];
            }

            numer += value * kernel;
            denom += kernel;
        }

        const auto result = aten::clamp(denom > 0.0F ? numer / denom : 1.0F, 0.0F, 1.0F);
        return result;
    }
}
}
