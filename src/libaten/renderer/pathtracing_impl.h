#pragma once

#include "defs.h"
#include "camera/camera.h"
#include "camera/pinhole.h"
#include "light/ibl.h"
#include "math/ray.h"
#include "renderer/aov.h"
#include "scene/scene.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

#include "renderer/pt_params.h"

namespace AT_NAME
{
    inline AT_DEVICE_MTRL_API void GeneratePath(
        aten::ray& generated_ray,
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t sample, uint32_t frame,
        AT_NAME::Path& paths,
        const aten::CameraParameter& camera,
        const uint32_t rnd)
    {
        paths.attrib[idx].isHit = false;

        if (paths.attrib[idx].isKill) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            0,
            scramble);

        float r1 = paths.sampler[idx].nextSample();
        float r2 = paths.sampler[idx].nextSample();

        float s = (ix + r1) / (float)(camera.width);
        float t = (iy + r2) / (float)(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        generated_ray = camsample.r;

        paths.throughput[idx].throughput = aten::vec3(1);
        paths.throughput[idx].pdfb = 1.0f;
        paths.attrib[idx].isTerminate = false;
        paths.attrib[idx].isSingular = false;

        paths.contrib[idx].samples += 1;
    }

    namespace _detail {
        template <typename A, typename B>
        inline AT_DEVICE_MTRL_API void AddVec3(A& dst, const B& add)
        {
            if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
                dst += add;
            }
            else {
                dst += make_float3(add.x, add.y, add.z);
            }
        }
    }

    template <typename AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_MTRL_API void ShadeMiss(
        int32_t idx,
        int32_t bounce,
        const aten::vec3& bg,
        AT_NAME::Path& paths,
        AOV_BUFFER_TYPE* aov_normal_depth = nullptr,
        AOV_BUFFER_TYPE* aov_albedo_meshid = nullptr)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            if (bounce == 0) {
                paths.attrib[idx].isKill = true;

                if (aov_normal_depth != nullptr && aov_albedo_meshid != nullptr)
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], bg);
                }
            }

            auto contrib = paths.throughput[idx].throughput * bg;
            _detail::AddVec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].isTerminate = true;
        }
    }

    template <typename AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_MTRL_API void ShadeMissWithEnvmap(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t bounce,
        int32_t envmap_idx,
        float envmapAvgIllum, float envmapMultiplyer,
        const AT_NAME::context& ctxt,
        const aten::CameraParameter& camera,
        AT_NAME::Path& paths,
        const aten::ray& ray,
        AOV_BUFFER_TYPE* aov_normal_depth = nullptr,
        AOV_BUFFER_TYPE* aov_albedo_meshid = nullptr)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
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

            auto uv = AT_NAME::envmap::convertDirectionToUV(dir);

#ifdef __CUDACC__
            // envmapidx is index to array of textures in context.
            // In GPU, sampleTexture requires texture id of CUDA. So, arguments is different.
            const auto bg = tex2D<float4>(ctxt.textures[envmap_idx], uv.x, uv.y);
#else
            const auto bg = AT_NAME::sampleTexture(envmap_idx, uv.x, uv.y, aten::vec4(1));
#endif
            auto emit = aten::vec3(bg.x, bg.y, bg.z);

            float misW = 1.0f;
            if (bounce == 0
                || (bounce == 1 && paths.attrib[idx].isSingular))
            {
                paths.attrib[idx].isKill = true;

                if (aov_normal_depth != nullptr && aov_albedo_meshid != nullptr)
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], bg);
                }
            }
            else {
                auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);

                emit *= envmapMultiplyer;
            }

            auto contrib = paths.throughput[idx].throughput * misW * emit;
            _detail::AddVec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].isTerminate = true;
        }
    }

    inline AT_DEVICE_MTRL_API void FillShadowRay(
        AT_NAME::ShadowRay& shadow_ray,
        const AT_NAME::context& ctxt,
        int32_t bounce,
        aten::sampler& sampler,
        const AT_NAME::PathThroughput& throughtput,
        const aten::MaterialParameter& mtrl,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        real hit_u, real hit_v,
        const aten::vec4& external_albedo,
        real pre_sampled_r = real(0))
    {
        shadow_ray.isActive = false;

        const auto lightnum = ctxt.get_light_num();

        if (lightnum <= 0 || mtrl.attrib.isSingular || mtrl.attrib.isTranslucent) {
            return;
        }

        const auto target_light_idx = aten::cmpMin<int32_t>(sampler.nextSample() * lightnum, lightnum - 1);
        const auto lightSelectPdf = 1.0f / lightnum;

        const auto& light = ctxt.GetLight(target_light_idx);

        bool isShadowRayActive = false;

        aten::LightSampleResult sampleres;
        AT_NAME::Light::sample(sampleres, light, ctxt, hit_pos, hit_nml, &sampler, bounce);

        const auto& posLight = sampleres.pos;
        const auto& nmlLight = sampleres.nml;
        real pdfLight = sampleres.pdf;

        auto dirToLight = normalize(sampleres.dir);
        auto distToLight = length(posLight - hit_pos);

        auto shadowRayOrg = hit_pos + AT_MATH_EPSILON * hit_nml;
        auto tmp = hit_pos + dirToLight - shadowRayOrg;
        auto shadowRayDir = normalize(tmp);

        shadow_ray.rayorg = shadowRayOrg;
        shadow_ray.raydir = shadowRayDir;
        shadow_ray.targetLightId = target_light_idx;
        shadow_ray.distToLight = distToLight;
        shadow_ray.lightcontrib = aten::vec3(0);
        {
            auto cosShadow = dot(hit_nml, dirToLight);

            real pdfb = AT_NAME::material::samplePDF(&mtrl, hit_nml, ray.dir, dirToLight, hit_u, hit_v);
            auto bsdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(&mtrl, hit_nml, ray.dir, dirToLight, hit_u, hit_v, external_albedo, pre_sampled_r);

            bsdf *= throughtput.throughput;

            // Get light color.
            auto emit = sampleres.finalColor;

            if (light.attrib.isInfinite || light.attrib.isSingular) {
                if (pdfLight > real(0) && cosShadow >= 0) {
                    auto misW = light.attrib.isSingular
                        ? 1.0f
                        : AT_NAME::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                    shadow_ray.lightcontrib =
                        (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;

                    isShadowRayActive = true;
                }
            }
            else {
                auto cosLight = dot(nmlLight, -dirToLight);

                if (cosShadow >= 0 && cosLight >= 0) {
                    auto dist2 = aten::squared_length(sampleres.dir);
                    auto G = cosShadow * cosLight / dist2;

                    if (pdfb > real(0) && pdfLight > real(0)) {
                        // Convert pdf from steradian to area.
                        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                        // p31 - p35
                        pdfb = pdfb * cosLight / dist2;

                        auto misW = AT_NAME::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);

                        shadow_ray.lightcontrib =
                            (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;

                        isShadowRayActive = true;
                    }
                }
            }
        }

        shadow_ray.isActive = isShadowRayActive;
    }

    namespace _detail {
        template <typename SCENE = void>
        inline AT_DEVICE_MTRL_API bool HitShadowRay(
            bool enableLod,
            const AT_NAME::context& ctxt,
            const AT_NAME::ShadowRay& shadowRay,
            SCENE* scene = nullptr)
        {
            auto targetLightId = shadowRay.targetLightId;
            auto distToLight = shadowRay.distToLight;

            const auto& light = ctxt.GetLight(targetLightId);
            const auto lightobj = (light.objid >= 0 ? &ctxt.GetObject(static_cast<uint32_t>(light.objid)) : nullptr);

            real distHitObjToRayOrg = AT_MATH_INF;

            // Ray aim to the area light.
            // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
            const aten::ObjectParameter* hitobj = lightobj;

            aten::Intersection isect;

            bool isHit = false;

            aten::ray r(shadowRay.rayorg, shadowRay.raydir);

            isHit = false;

            if constexpr (!std::is_void_v<std::remove_pointer_t<SCENE>>) {
                // NOTE:
                // operation has to be related with template arg SCENE.
                if (scene) {
                    aten::hitrecord rec;
                    isHit = scene->hit(ctxt, r, AT_MATH_EPSILON, distToLight - AT_MATH_EPSILON, rec, isect);
                }
            }
            else {
                isHit = intersectCloser(&ctxt, r, &isect, distToLight - AT_MATH_EPSILON, enableLod);
            }

            if (isHit) {
                hitobj = &ctxt.GetObject(static_cast<uint32_t>(isect.objid));
            }

            isHit = AT_NAME::scene::hitLight(
                isHit,
                light.attrib,
                lightobj,
                distToLight,
                distHitObjToRayOrg,
                isect.t,
                hitobj);

            return isHit;
        }
    }

    template <typename SCENE = void>
    inline AT_DEVICE_MTRL_API void HitShadowRay(
        int32_t idx,
        int32_t bounce,
        const AT_NAME::context& ctxt,
        AT_NAME::Path paths,
        const AT_NAME::ShadowRay* shadow_rays,
        SCENE* scene = nullptr)
    {
        if (paths.attrib[idx].isKill || paths.attrib[idx].isTerminate) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        const auto& shadowRay = shadow_rays[idx];

        if (!shadowRay.isActive) {
            return;
        }

        // TODO
        bool enableLod = (bounce >= 2);

        aten::hitrecord tmpRec;

        auto isHit = _detail::HitShadowRay<std::remove_pointer_t<decltype(scene)>>(
            enableLod, ctxt, shadowRay, scene);

        if (isHit) {
            _detail::AddVec3(paths.contrib[idx].contrib, shadowRay.lightcontrib);
        }
    }

    inline AT_DEVICE_MTRL_API bool HitImplicitLight(
        bool is_back_facing,
        int32_t bounce,
        AT_NAME::PathContrib& path_contrib,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        float hit_area,
        const aten::MaterialParameter& hit_target_mtrl)
    {
        if (!hit_target_mtrl.attrib.isEmissive) {
            return false;
        }

        if (is_back_facing) {
            return false;
        }

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

        auto contrib = path_throughput.throughput * weight * static_cast<aten::vec3>(hit_target_mtrl.baseColor);
        _detail::AddVec3(path_contrib.contrib, contrib);

        // When ray hit the light, tracing will finish.
        path_attrib.isTerminate = true;
        return true;
    }
}
