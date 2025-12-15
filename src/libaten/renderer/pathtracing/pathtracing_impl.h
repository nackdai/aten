#pragma once

#include <optional>

#include "defs.h"
#include "camera/camera.h"
#include "camera/pinhole.h"
#include "geometry/EvaluateHitResult.h"
#include "light/ibl.h"
#include "light/light_impl.h"
#include "math/ray.h"
#include "material/material.h"
#include "material/material_impl.h"
#include "misc/tuple.h"
#include "misc/type_traits.h"
#include "renderer/aov.h"
#include "scene/scene.h"

#include "accelerator/threaded_bvh_traverser.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

#include "renderer/pathtracing/pt_params.h"
#include "renderer/pathtracing/pathtracing_nee_impl.h"

#include "math/cuda_host_common_math.h"

namespace AT_NAME
{
    namespace _detail_pathtracing_impl {
        inline AT_DEVICE_API aten::vec3 ApplyAlphaBlend(
            const aten::vec3& c,
            const PathThroughput& throughput)
        {
            return throughput.alpha_blend.transmission * c + throughput.alpha_blend.throughput;
        }

        inline AT_DEVICE_API void ClearAlphaBlend(
            PathThroughput& throughput,
            PathAttribute& attrib)
        {
            throughput.alpha_blend.transmission = 1.0F;
            throughput.alpha_blend.throughput = aten::vec3(0.0F);
            attrib.attr.is_accumulating_alpha_blending = false;
        }
    }

    inline AT_DEVICE_API void ClearPathAttribute(PathAttribute& attrib)
    {
        attrib.attr.isHit = false;
        attrib.attr.is_terminated = false;
        attrib.attr.is_singular = false;
        attrib.attr.will_update_depth = true;
        attrib.attr.does_use_throughput_depth = false;
        attrib.attr.is_accumulating_alpha_blending = true;
        attrib.last_hit_mtrl_idx = -1;
    }

    inline AT_DEVICE_API void GeneratePath(
        aten::ray& generated_ray,
        int32_t idx,
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        int32_t sample, uint32_t frame,
        AT_NAME::Path& paths,
        const aten::CameraParameter& camera,
        const uint32_t rnd)
    {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            0,
            scramble);
#else
        static_assert(false, "Other samplers are not supported");
#endif

        float r1 = paths.sampler[idx].nextSample();
        float r2 = paths.sampler[idx].nextSample();

        float s = (ix + r1) / (float)(camera.width);
        float t = (iy + r2) / (float)(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        generated_ray = camsample.r;

        paths.throughput[idx].throughput = aten::vec3(1);
        paths.throughput[idx].pdfb = 1.0f;
        paths.throughput[idx].medium.depth_count = 0;
        paths.throughput[idx].medium.stack.clear();

        _detail_pathtracing_impl::ClearAlphaBlend(paths.throughput[idx], paths.attrib[idx]);

        ClearPathAttribute(paths.attrib[idx]);
        paths.attrib[idx].screen_space_x = ix;
        paths.attrib[idx].screen_space_y = iy;

        paths.contrib[idx].samples += 1;
    }

    template <class AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_API void ShadeMiss(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t bounce,
        const AT_NAME::context& ctxt,
        const aten::CameraParameter& camera,
        AT_NAME::Path& paths,
        const aten::ray& ray,
        aten::span<AOV_BUFFER_TYPE> aov_normal_depth = nullptr,
        aten::span<AOV_BUFFER_TYPE> aov_albedo_meshid = nullptr)
    {
        bounce = paths.attrib[idx].attr.does_use_throughput_depth
            ? paths.throughput[idx].medium.depth_count
            : bounce;

        if (!paths.attrib[idx].attr.is_terminated && !paths.attrib[idx].attr.isHit) {
            aten::vec3 dir = ray.dir;

            if (bounce == 0) {
                // Suppress jittering envrinment map.
                // So, re-sample ray without random.

                // TODO
                // More efficient way...

                float s = ix / (float)(width);
                float t = iy / (float)(height);

                AT_NAME::CameraSampleResult camsample;
                AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

                dir = camsample.r.dir;
            }

            auto emit = AT_NAME::Background::SampleFromRay(dir, ctxt.scene_rendering_config.bg, ctxt);

            float misW = 1.0f;
            if (bounce == 0
                || (bounce == 1 && paths.attrib[idx].attr.is_singular))
            {
                if (!aov_normal_depth.empty() && !aov_albedo_meshid.empty())
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], emit);
                }
            }
            else {
                auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, ctxt.scene_rendering_config.bg.avgIllum);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }

            auto contrib = _detail_pathtracing_impl::ApplyAlphaBlend(misW * emit, paths.throughput[idx]);
            contrib *= paths.throughput[idx].throughput;

            aten::AddVec3(paths.contrib[idx].contrib, contrib);

            _detail_pathtracing_impl::ClearAlphaBlend(paths.throughput[idx], paths.attrib[idx]);

            paths.attrib[idx].attr.is_terminated = true;
        }
    }

    inline AT_DEVICE_API aten::tuple<aten::LightSampleResult, float, int32_t> SampleLight(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& mtrl,
        int32_t bounce,
        aten::sampler& sampler,
        const aten::vec3& org,
        const aten::vec3& nml,
        bool is_on_surface = true)
    {
        aten::LightSampleResult light_sample;
        float light_select_prob = 0.0F;
        int target_light_idx = -1;

        const auto lightnum = static_cast<int32_t>(ctxt.GetLightNum());

        bool is_invalid_mtrl = is_on_surface && (mtrl.attrib.is_singular || mtrl.attrib.is_translucent);

        if (lightnum <= 0 || is_invalid_mtrl) {
            return aten::make_tuple(light_sample, light_select_prob, target_light_idx);
        }

        target_light_idx = aten::min<decltype(lightnum)>(
            static_cast<decltype(lightnum)>(sampler.nextSample() * lightnum), lightnum - 1);
        light_select_prob = 1.0f / lightnum;

        const auto& light = ctxt.GetLight(target_light_idx);
        AT_NAME::Light::sample(light_sample, light, ctxt, org, nml, &sampler, bounce);

        return aten::make_tuple(light_sample, light_select_prob, target_light_idx);
    }

    inline AT_DEVICE_API void FillShadowRay(
        const int32_t idx,
        aten::ShadowRay& shadow_ray,
        const AT_NAME::context& ctxt,
        int32_t bounce,
        const AT_NAME::Path& paths,
        const aten::MaterialParameter& mtrl,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        float hit_u, float hit_v,
        const aten::vec4& external_albedo,
        float pre_sampled_r = float(0))
    {
        aten::sampler& sampler = paths.sampler[idx];

        shadow_ray.isActive = false;

        aten::LightSampleResult sampleres;
        float lightSelectPdf = 0.0F;
        int32_t target_light_idx = -1;
        aten::tie(sampleres, lightSelectPdf, target_light_idx) = SampleLight(ctxt, mtrl, bounce, sampler, hit_pos, hit_nml);

        if (target_light_idx < 0) {
            return;
        }

        bool isShadowRayActive = false;

        const auto& posLight = sampleres.pos;

        auto dirToLight = normalize(sampleres.dir);
        auto distToLight = length(posLight - hit_pos);

        auto shadowRayOrg = aten::ray::Offset(hit_pos, hit_nml);
        auto shadowRayDir = dirToLight;

        shadow_ray.rayorg = shadowRayOrg;
        shadow_ray.raydir = shadowRayDir;
        shadow_ray.targetLightId = target_light_idx;
        shadow_ray.distToLight = distToLight;
        shadow_ray.lightcontrib = aten::vec3(0);

        auto radiance = ComputeRadianceNEE(
            ctxt, paths.throughput[idx].throughput,
            ray.dir, hit_nml,
            mtrl, pre_sampled_r, hit_u, hit_v,
            lightSelectPdf, sampleres);
        if (radiance.has_value()) {
            const auto& r = radiance.value();
            shadow_ray.lightcontrib = paths.throughput[idx].throughput * r * external_albedo;
            isShadowRayActive = true;
        }

        shadow_ray.isActive = isShadowRayActive;
    }

    inline AT_DEVICE_API bool HitTestToTargetLight(
        const AT_NAME::context& ctxt,
        const aten::ray& original_ray,
        const aten::LightParameter& target_light,
        const float distance_to_target_light,
        const aten::MaterialParameter& surface_mtrl
    )
    {
        bool is_hit_to_light = false;

        const auto target_light_obj = (target_light.IsValidLightObjectId()
            ? &ctxt.GetObject(static_cast<uint32_t>(target_light.arealight_objid))
            : nullptr);

        float distHitObjToRayOrg = AT_MATH_INF;
        const auto distToLight = distance_to_target_light;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::ObjectParameter* hitobj = target_light_obj;

        aten::ray r{ original_ray };

        // At least one hit test to the light is necessary.
        size_t max_lookups = ctxt.scene_rendering_config.enable_alpha_blending ? 10 : 1;

        const bool need_stencil_check = surface_mtrl.stencil_type == aten::StencilType::ALWAYS;
        max_lookups = need_stencil_check ? 10 : max_lookups;

        // NOTE:
        // For safety, jump out the while loop forcibly.
        for (size_t i = 0; i < max_lookups; i++) {
            aten::Intersection isect;

            bool isHit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                r,
                AT_MATH_EPSILON, distToLight - AT_MATH_EPSILON);

            if (isHit) {
                hitobj = &ctxt.GetObject(static_cast<uint32_t>(isect.objid));

                aten::hitrecord rec;
                AT_NAME::evaluate_hit_result(rec, *hitobj, ctxt, r, isect);

                const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

                bool is_ignore_hit = false;

                if (need_stencil_check) {
                    if (mtrl.stencil_type == aten::StencilType::STENCIL) {
                        is_ignore_hit = true;
                    }
                }
                if (AT_NAME::material::isTranslucentByAlpha(ctxt, mtrl, rec.u, rec.v)) {
                    is_ignore_hit = true;
                }

                if (is_ignore_hit) {
                    auto orienting_normal = rec.normal;
                    const bool is_same_facing = dot(rec.normal, original_ray.dir) > 0.0F;
                    if (!is_same_facing) {
                        orienting_normal = -orienting_normal;
                    }

                    // To go path through the object, specfy the oppoiste normal.
                    r = aten::ray(rec.p, original_ray.dir, orienting_normal);
                    continue;
                }
            }

            is_hit_to_light = AT_NAME::scene::hitLight(
                isHit,
                target_light.attrib,
                target_light_obj,
                distToLight,
                distHitObjToRayOrg,
                isect.t,
                hitobj);

            break;
        }

        return is_hit_to_light;
    }

    inline AT_DEVICE_API bool HitShadowRay(
        int32_t bounce,
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& surface_mtrl,
        const AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathContrib& path_contrib,
        const aten::ShadowRay& shadow_ray
    )
    {
        if (path_attrib.attr.is_terminated) {
            return false;
        }

        if (!shadow_ray.isActive) {
            return false;
        }

        // TODO
        bool enableLod = (bounce >= 2);

        const auto targetLightId = shadow_ray.targetLightId;
        const auto distToLight = shadow_ray.distToLight;

        const auto& light = ctxt.GetLight(targetLightId);

        // NOTE:
        // Offset is already applied to shadow_ray.rayorg in FillShadowRay.
        aten::ray r(shadow_ray.rayorg, shadow_ray.raydir);

        const auto is_hit_to_light = HitTestToTargetLight(
            ctxt, r,
            light, distToLight,
            surface_mtrl
        );

        if (is_hit_to_light) {
            aten::AddVec3(path_contrib.contrib, shadow_ray.lightcontrib);
        }

        return is_hit_to_light;
    }

    inline AT_DEVICE_API bool HitImplicitLight(
        const AT_NAME::context& ctxt,
        int32_t hit_obj_id,
        bool is_back_facing,
        int32_t bounce,
        AT_NAME::PathContrib& path_contrib,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::hitrecord& hrec,
        const aten::MaterialParameter& hit_target_mtrl)
    {
        if (!hit_target_mtrl.attrib.is_emissive) {
            return false;
        }

        if (is_back_facing) {
            return false;
        }

        const aten::vec3& hit_pos = hrec.p;
        const aten::vec3& hit_nml = hrec.normal;
        float hit_area = hrec.area;

        const auto& obj = ctxt.GetObject(hit_obj_id);
        const auto& light = ctxt.GetLight(obj.light_id);
        const auto light_color = AT_NAME::AreaLight::ComputeLightColor(light, hit_area);

        float weight = 1.0f;

        if (bounce > 0) {
            auto cosLight = dot(hit_nml, -ray.dir);
            auto dist2 = aten::squared_length(hit_pos - ray.org);

            if (cosLight >= 0) {
                auto pdfLight = 1 / hit_area;

                // If logic comes here, light has area.
                // But, sampling is fully based on solid angle (path) because the brdf sampling comes from the previous bounce and it's solid angle base sampling.
                // So, area PDF has to be converted to solid angle PDF.
                pdfLight = pdfLight * dist2 / cosLight;

                weight = _detail::ComputeBalanceHeuristic(path_throughput.pdfb, pdfLight);
            }
        }

        // NOTE:
        // In the previous bounce, (bsdf * cos / path_pdf) has been computed and multiplied to path_throughput.throughput.
        // Therefore, no need to compute it again here.

        auto contrib{ path_throughput.throughput * weight * light_color };
        aten::AddVec3(path_contrib.contrib, contrib);

        // When ray hit the light, tracing will finish.
        path_attrib.attr.is_terminated = true;
        return true;
    }

    inline AT_DEVICE_API bool HitTeminatedMaterial(
        const AT_NAME::context& ctxt,
        aten::sampler& sampler,
        int32_t hit_obj_id,
        bool is_back_facing,
        int32_t bounce,
        AT_NAME::PathContrib& path_contrib,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::hitrecord& hrec,
        const aten::vec3& external_albedo,
        const aten::MaterialParameter& hit_target_mtrl)
    {
        const auto is_teminate_mtrl = material::IsTerminatedMaterial(hit_target_mtrl);
        if (!is_teminate_mtrl) {
            return false;
        }

        const auto mtrl_type = hit_target_mtrl.type;

        if (mtrl_type == aten::MaterialType::Emissive) {
            // Implicit conection to light.
            return HitImplicitLight(
                ctxt,
                hit_obj_id, is_back_facing, bounce,
                path_contrib, path_attrib, path_throughput,
                ray, hrec, hit_target_mtrl);
        }
        else if (mtrl_type == aten::MaterialType::Toon
                || mtrl_type == aten::MaterialType::StylizedBrdf)
        {
            if (bounce == 0) {
                // Treat toon as a light.
                const auto toon_bsdf = Toon::bsdf(
                    ctxt, hit_target_mtrl,
                    path_throughput, path_attrib,
                    sampler,
                    hrec.p, hrec.normal, ray.dir,
                    hrec.u, hrec.v
                );

                // NOTE:
                // The accumulated alpha blending color is applied in Toon::bsdf.
                // So, no need to do it here.

                aten::AddVec3(path_contrib.contrib, path_throughput.throughput * toon_bsdf * external_albedo);

                path_attrib.attr.is_terminated = true;
            }
            return true;
        }
        else {
            AT_ASSERT(false);
            return false;
        }
    }

    inline AT_DEVICE_API bool CheckMaterialTranslucentByAlpha(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& mtrl,
        float u, float v,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        aten::ray& ray,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput)
    {
        // TODO:
        // How to deal with the alpha material on the other hand of the translucent material like refraction.

        if (!ctxt.scene_rendering_config.enable_alpha_blending) {
            return false;
        }

        if (!path_attrib.attr.is_accumulating_alpha_blending) {
            return false;
        }

        // If the material itself is originally translucent, we don't care alpha translucency.
        if (mtrl.attrib.is_translucent) {
            return false;
        }

        // NOTE:
        // material::getTranslucentAlpha returns the multiplied alpha.
        // Why we call sampleTexture to get alpha value instead of calling material::getTranslucentAlpha is retrieving albedo color and alpha in the same time.
        // Therefore, we need to compute the multiplied alpha value.
        const auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, u, v, aten::vec4(1.0F));
        const auto alpha = albedo.a * mtrl.baseColor.a;

        if (alpha < 1.0F)
        {
            // Just through the object.
            // NOTE:
            // Ray go through to the opposite direction. So, we need to specify inverted normal.
            // Even if accumulating alpha blending already has been stopped but the hit object has alpha translucent material,
            // we skip the following shadeing sequence.
            // It means the ray go through to the opposite direction as if the hit object is ignored.
            ray = aten::ray(hit_pos, ray.dir, -hit_nml);

            const aten::vec3 alpha_belended_radiance{
                path_throughput.alpha_blend.transmission * mtrl.baseColor * albedo * alpha
            };
            aten::AddVec3(path_throughput.alpha_blend.throughput, alpha_belended_radiance);
            path_throughput.alpha_blend.transmission *= (1.0F - alpha);

            path_attrib.attr.is_singular = true;
            return true;
        }
        else {
            // If ray hits to non-alpha face, terminate to accumulate alpha transmission.
            path_attrib.attr.is_accumulating_alpha_blending = false;
        }

        return false;
    }

    inline AT_DEVICE_API aten::ray AdvanceAlphaBlendPath(
        const AT_NAME::context& ctxt,
        const aten::ray& original_ray,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput
    )
    {
        if (!path_attrib.attr.is_accumulating_alpha_blending) {
            return original_ray;
        }

        auto ray{ original_ray };

        // TODO
        constexpr size_t MAX_LOOKUPS = 10;

        size_t loop_count = 0;

        for (loop_count = 0; loop_count < MAX_LOOKUPS; loop_count++) {
            aten::Intersection isect;
            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect, ctxt, ray,
                AT_MATH_EPSILON, AT_MATH_INF
            );

            if (is_hit) {
                const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

                aten::hitrecord rec;
                AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

                const auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
                    ctxt,
                    ctxt.GetMaterial(rec.mtrlid),
                    rec.u, rec.v, rec.p, rec.normal,
                    ray,
                    path_attrib, path_throughput);
                if (!is_translucent_by_alpha) {
                    return ray;
                }
            }
            else {
                path_attrib.attr.is_accumulating_alpha_blending = false;
                return ray;
            }
        }

        path_attrib.attr.is_accumulating_alpha_blending = false;
        return original_ray;
    }

    inline AT_DEVICE_API bool CheckStencil(
        aten::ray& curr_ray,
        AT_NAME::PathAttribute& path_attrib,
        const int32_t bounce,
        const AT_NAME::context& ctxt,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        const aten::MaterialParameter& mtrl
    )
    {
        if (bounce > 0
            || mtrl.stencil_type != aten::StencilType::STENCIL
        ) {
            return false;
        }

        constexpr size_t MAX_LOOKUPS = 10;

        // Just through the object.
        // Ray go through to the opposite direction. So, we need to specify inverted normal.
        aten::ray ray(hit_pos, curr_ray.dir, -hit_nml);

        for (size_t i = 0; i < MAX_LOOKUPS; i++) {
            aten::Intersection isect;
            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect, ctxt, ray,
                AT_MATH_EPSILON, AT_MATH_INF
            );

            if (is_hit) {
                const auto& hit_mtrl = ctxt.GetMaterial(isect.mtrlid);
                if (hit_mtrl.stencil_type == aten::StencilType::ALWAYS) {
                    // TODO:
                    // Unfortunately, it's hard to replace the current original evaludate resulted before stencil check.
                    // So, just update ray expected to hit the none stencil object which is found here.
                    curr_ray = ray;
                    return true;
                }
                else if (hit_mtrl.stencil_type == aten::StencilType::NONE) {
                    return false;
                }
                else {
                    // Prepare for next.
                    aten::hitrecord rec;
                    const auto& obj = ctxt.GetObject(isect.objid);
                    AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

                    // Just through the object.
                    // Ray go through to the opposite direction. So, we need to specify inverted normal.
                    ray = aten::ray(rec.p, ray.dir, -rec.normal);
                }
            }
        }

        return false;
    }

    inline AT_DEVICE_API float ComputeRussianProbability(
        int32_t bounce,
        int32_t rr_bounce,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        aten::sampler& sampler)
    {
        float russian_prob = 1.0f;

        if (bounce > rr_bounce) {
            if (aten::squared_length(path_throughput.throughput) > 0) {
                russian_prob = aten::max_from_vec3(path_throughput.throughput);
                auto p = sampler.nextSample();
                path_attrib.attr.is_terminated = (p >= russian_prob);
            }
        }

        return russian_prob;
    }

    inline AT_DEVICE_API void PrepareForNextBounce(
        int32_t idx,
        const aten::hitrecord& rec,
        bool is_backfacing,
        float russian_prob,
        const aten::vec3& normal,
        const aten::MaterialParameter& mtrl,
        const AT_NAME::MaterialSampling& sampling,
        const aten::vec3& albedo,
        AT_NAME::Path& paths,
        aten::ray* rays)
    {
        const auto next_dir = normalize(sampling.dir);
        const auto pdfb = sampling.pdf;
        const auto bsdf = sampling.bsdf;

        auto ray_along_normal = normal;

        // Adjust the surface normal to along with the same direction as the next ray.
        ray_along_normal = dot(ray_along_normal, next_dir) >= 0.0f ? ray_along_normal : -ray_along_normal;

        auto c = dot(ray_along_normal, static_cast<aten::vec3>(next_dir));

        if (pdfb > 0 && c > 0) {
            paths.throughput[idx].throughput *= albedo * bsdf * c / pdfb;
            paths.throughput[idx].throughput /= russian_prob;
        }
        else {
            paths.attrib[idx].attr.is_terminated = true;
        }

        _detail_pathtracing_impl::ClearAlphaBlend(paths.throughput[idx], paths.attrib[idx]);

        if (paths.attrib[idx].attr.is_terminated) {
            return;
        }

        paths.throughput[idx].pdfb = pdfb;
        paths.attrib[idx].attr.is_singular = mtrl.attrib.is_singular;
        paths.attrib[idx].last_hit_mtrl_idx = mtrl.id;

        // Make next ray.
        rays[idx] = aten::ray(rec.p, next_dir, ray_along_normal);
    }

    template <bool ENABLE_ALPHA_TRANLUCENT = false>
    inline AT_DEVICE_API void ShadePathTracing(
        int32_t idx,
        const AT_NAME::context& ctxt,
        AT_NAME::Path paths,
        aten::ray* rays,
        const aten::Intersection& isect,
        aten::MaterialParameter& mtrl,
        aten::ShadowRay& shadow_ray,
        int32_t rrDepth,
        int32_t bounce)
    {
        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& obj = ctxt.GetObject(isect.objid);

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        aten::vec3 orienting_normal = rec.normal;

        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        shadow_ray.isActive = false;

        // Implicit conection to light.
        auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
            ctxt, isect.objid,
            isBackfacing,
            bounce,
            paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
            ray,
            rec,
            mtrl);
        if (is_hit_implicit_light) {
            return;
        }

        if (!mtrl.attrib.is_translucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        auto pre_sampled_r = material::applyNormal(
            ctxt,
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, sampler);

        if constexpr (ENABLE_ALPHA_TRANLUCENT) {
            // Check transparency or translucency.
            auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
                ctxt,
                mtrl,
                rec.u, rec.v, rec.p,
                orienting_normal,
                rays[idx],
                paths.attrib[idx],
                paths.throughput[idx]);
            if (is_translucent_by_alpha) {
                return;
            }
        }

        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            idx,
            shadow_ray,
            ctxt,
            bounce,
            paths,
            mtrl,
            ray,
            rec.p, orienting_normal,
            rec.u, rec.v, albedo,
            pre_sampled_r);

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);

        AT_NAME::MaterialSampling sampling;
        material::sampleMaterial(
            &sampling,
            ctxt,
            paths.throughput[idx].throughput,
            &mtrl,
            orienting_normal,
            ray.dir,
            rec.normal,
            sampler, pre_sampled_r,
            rec.u, rec.v);

        AT_NAME::PrepareForNextBounce(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            mtrl, sampling,
            albedo,
            paths, rays);
    }
}
