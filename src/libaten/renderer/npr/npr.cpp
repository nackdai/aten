#include <array>

#include "accelerator/threaded_bvh_traverser.h"
#include "material/material_impl.h"
#include "misc/omputil.h"
#include "renderer/ao/aorenderer_impl.h"
#include "renderer/npr/npr_pathtracer.h"
#include "renderer/npr/npr_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void NprPathTracer::radiance(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera,
        aten::hitrecord* first_hrec/*= nullptr*/)
    {
        int32_t depth = 0;

        while (depth < m_maxDepth) {
            bool willContinue = true;
            auto& isect = isects_[idx];

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].attr.isHit = false;

            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                if (depth == 0 && first_hrec) {
                    const auto& obj = ctxt.GetObject(isect.objid);
                    AT_NAME::evaluate_hit_result(*first_hrec, obj, ctxt, ray, isect);
                }

                path_host_.paths.attrib[idx].attr.isHit = true;

                AdvanceNPRPath(idx, ctxt, path_host_.paths, rays_, isect);

                shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, m_rrDepth, depth);

                AT_NAME::AdvanceAlphaBlendPath(
                    ctxt, rays_[idx],
                    path_host_.paths.attrib[idx], path_host_.paths.throughput[idx]);

                const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

                if (depth == 0) {
                    HitShadowRayWithKeepingIfHitToLight(
                        idx, depth,
                        ctxt, path_host_.paths, isect,
                        shadow_rays_);
                }
                else {
                    std::ignore = AT_NAME::HitShadowRay(
                        depth,
                        ctxt, mtrl,
                        path_host_.paths.attrib[idx],
                        path_host_.paths.contrib[idx],
                        shadow_rays_[idx]);
                }

                willContinue = !path_host_.paths.attrib[idx].attr.is_terminated;
            }
            else {
                ShadeMiss(
                    idx,
                    ix, iy,
                    width, height,
                    depth,
                    ctxt, camera,
                    path_host_.paths, rays_[idx]);

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }

    void NprPathTracer::radiance_with_feature_line(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera)
    {
        const auto pixel_width = Camera::ComputePixelWidthAtDistance(camera, 1);
        const float feature_line_width{ ctxt.scene_rendering_config.feature_line.line_width };

        auto& sample_ray_info = feature_line_sample_ray_infos_[idx];
        auto* sampler = &path_host_.paths.sampler[idx];
        const auto& ray = rays_[idx];

        constexpr auto SampleRayNum = std::remove_reference_t< decltype(sample_ray_info)>::size;

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<SampleRayNum>(
            sample_ray_info.descs, sample_ray_info.disc,
            ray, *sampler,
            feature_line_width, pixel_width);

        int32_t depth = 0;

        while (depth < m_maxDepth) {
            bool willContinue = true;
            Intersection isect;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].attr.isHit = false;

            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            constexpr auto SampleRayNum = decltype(feature_line_sample_ray_infos_)::value_type::size;

            if (is_hit) {
                AT_NAME::npr::ShadeSampleRay<SampleRayNum>(
                    pixel_width,
                    idx, depth,
                    ctxt, camera,
                    ray, isect,
                    path_host_.paths,
                    feature_line_sample_ray_infos_.data()
                );

                path_host_.paths.attrib[idx].attr.isHit = true;

                AdvanceNPRPath(idx, ctxt, path_host_.paths, rays_, isect);

                shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, m_rrDepth, depth);

                const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

                std::ignore = AT_NAME::HitShadowRay(
                    depth,
                    ctxt, mtrl,
                    path_host_.paths.attrib[idx],
                    path_host_.paths.contrib[idx],
                    shadow_rays_[idx]);

                willContinue = !path_host_.paths.attrib[idx].attr.is_terminated;
            }
            else {
                AT_NAME::npr::ShadeMissSampleRay<SampleRayNum>(
                    pixel_width,
                    idx, depth,
                    ctxt,
                    ray,
                    path_host_.paths,
                    feature_line_sample_ray_infos_.data()
                );

                ShadeMiss(
                    idx,
                    ix, iy,
                    width, height,
                    depth,
                    ctxt, camera,
                    path_host_.paths, rays_[idx]);

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }

    void NprPathTracer::AdvanceNPRPath(
        const int32_t idx,
        const aten::context& ctxt,
        const aten::Path& paths,
        std::vector<aten::ray>& rays,
        aten::Intersection& isect)
    {
        const auto& ray = rays[idx];

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

        // �����ʒu�̖@��.
        // ���̂���̃��C�̓��o���l��.
        aten::vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid,
            rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, 0);

        // Apply normal map.
        int32_t normalMap = mtrl.normalMap;
        auto pre_sampled_r = AT_NAME::material::applyNormal(
            ctxt,
            &mtrl,
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths.sampler[idx]);

        bool need_advance_path_one_more = false;

        // Check stencil.
        auto is_stencil = AT_NAME::CheckStencil(
            rays[idx], paths.attrib[idx],
            0,
            ctxt,
            rec.p, orienting_normal,
            mtrl
        );
        if (is_stencil) {
            need_advance_path_one_more = true;
        }
        else if (ctxt.scene_rendering_config.enable_alpha_blending) {
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
                rays[idx] = AT_NAME::AdvanceAlphaBlendPath(
                    ctxt, rays[idx],
                    paths.attrib[idx], paths.throughput[idx]);

                need_advance_path_one_more = true;
            }
        }

        if (need_advance_path_one_more) {
            bool isHit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect, ctxt, rays[idx],
                AT_MATH_EPSILON, AT_MATH_INF);

            paths.attrib[idx].attr.isHit = isHit;
        }
    }

    void NprPathTracer::HitShadowRayWithKeepingIfHitToLight(
        int32_t idx, int32_t bounce,
        const aten::context& ctxt,
        const aten::Path& paths,
        const aten::Intersection& isect,
        std::vector<aten::ShadowRay>& shadowRays)
    {
        auto mtrl = ctxt.GetMaterial(isect.mtrlid);

        const auto original_mtrl_type = mtrl.type;

        const auto is_toon_material = (original_mtrl_type == aten::MaterialType::Toon
            || original_mtrl_type == aten::MaterialType::StylizedBrdf);

        if (is_toon_material) {
            // Replace toon material to lambertian for shadow ray test.
            mtrl.type = aten::MaterialType::Diffuse;
        }

        auto& shadow_ray = shadowRays[idx];

        // Reset termination flag to trace shadow ray forcibly.
        auto path_attrib = paths.attrib[idx];
        path_attrib.attr.is_terminated = false;

        AT_NAME::PathContrib path_contrib;

        // If material is toon material,
        // the contribution from shadow ray should not be applied to the rendering result.
        const auto is_hit_to_light = AT_NAME::HitShadowRay(
            bounce,
            ctxt, mtrl,
            path_attrib,
            is_toon_material ? path_contrib : paths.contrib[idx],
            shadow_ray);

        // For latter filtering, keep shadow ray if it hits to light.
        shadow_ray.isActive = is_hit_to_light;
    }

    template <class SrcType>
    void NprPathTracer::ApplyBilateralFilter(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const std::vector<aten::Intersection>& isects,
        const SrcType* src,
        std::function<void(float)> dst)
    {
        float filtered_color = 1.0F;

        constexpr int32_t KernelSizeH = 5;
        constexpr bool IsHorizontal = true;

        filtered_color = AT_NAME::ao::ApplyBilateralFilter<SrcType, float, IsHorizontal, KernelSizeH>(
            ix, iy,
            width, height,
            2.0F, 2.0F,
            src, isects.data()
        );

        if (filtered_color > 0 && filtered_color < 1) {
            int xxx = 0;
        }

        dst(filtered_color);
    }

    void NprPathTracer::OnRender(
        context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;
        m_rrDepth = dst.russianRouletteDepth;

        if (m_rrDepth > m_maxDepth) {
            m_rrDepth = m_maxDepth - 1;
        }

        if (rays_.empty()) {
            rays_.resize(width * height);
        }
        if (shadow_rays_.empty()) {
            shadow_rays_.resize(width * height);
        }

        if (ctxt.scene_rendering_config.feature_line.enabled) {
            if (feature_line_sample_ray_infos_.size() == 0) {
                feature_line_sample_ray_infos_.resize(width * height);
            }
        }
        if (isects_.empty()) {
            isects_.resize(width * height);
        }
        if (contributes_.empty()) {
            contributes_.resize(width * height);
        }

        path_host_.init(width, height);
        path_host_.Clear(GetFrameCount());

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto thread_idx = OMPUtil::getThreadIdx();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif
                    int32_t idx = y * width + x;

                    const auto curr_sample = 0;

                    std::ignore = RenderPerSample(
                        x, y, width, height, curr_sample,
                        scene, ctxt, camera);

                    const auto& c = path_host_.paths.contrib[idx].contrib;

                    if (!isInvalidColor(c)) {
                        contributes_[idx] += aten::vec4(c, 1.0F);

                        if (samples == 1) {
                            auto curr_contrib = contributes_[idx];
                            curr_contrib /= curr_contrib.w;
                            dst.buffer->put(x, y, vec4(curr_contrib.x, curr_contrib.y, curr_contrib.z, 1));
                        }
                    }
                }
            }

            if (samples >= 1) {
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp single
#endif
                if (ctxt.screen_space_texture.empty()) {
                    ctxt.screen_space_texture.init(width, height, 1);
                }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                for (int32_t y = 0; y < height; y++) {
                    for (int32_t x = 0; x < width; x++) {
                        ApplyBilateralFilter(
                            x, y, width, height, isects_, shadow_rays_.data(),
                            [&ctxt, x, y](float fitered_color) {
                                ctxt.screen_space_texture.PutByXYcoord(x, y, fitered_color);
                            });
                    }
                }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                for (int32_t y = 0; y < height; y++) {
                    for (int32_t x = 0; x < width; x++) {
                        int32_t idx = y * width + x;

                        for (int32_t i = 0; i < samples; i++) {
#ifdef RELEASE_DEBUG
                            if (x == BREAK_X && y == BREAK_Y) {
                                DEBUG_BREAK();
                            }
#endif

                            const auto is_continue = RenderPerSample(
                                x, y, width, height, i,
                                scene, ctxt, camera);

                            const auto& c = path_host_.paths.contrib[idx].contrib;

                            if (!isInvalidColor(c)) {
                                contributes_[idx] += aten::vec4(c, 1.0F);
                            }

                            if (!is_continue) {
                                break;
                            }
                        }

                        auto curr_contrib = contributes_[idx];
                        curr_contrib /= curr_contrib.w;
                        dst.buffer->put(x, y, vec4(curr_contrib.x, curr_contrib.y, curr_contrib.z, 1));
                    }
                }
            }
        }
    }

    bool NprPathTracer::RenderPerSample(
        int32_t x, int32_t y,
        int32_t width, int32_t height,
        int32_t sample,
        aten::scene* scene,
        const aten::context& ctxt,
        const aten::Camera* camera)
    {
        const auto idx = y * width + x;

        const auto rnd = aten::getRandom(idx);
        const auto& camsample = camera->param();

        GeneratePath(
            rays_[idx],
            idx,
            x, y, width, height,
            sample, GetFrameCount(),
            path_host_.paths,
            camsample,
            rnd);

        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

        if (ctxt.scene_rendering_config.feature_line.enabled) {
            radiance_with_feature_line(
                idx,
                x, y, width, height,
                ctxt, scene, camsample);
        }
        else {
            radiance(
                idx,
                x, y, width, height,
                ctxt, scene, camsample);
        }

        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, sample);
            return true;
        }

        auto c = path_host_.paths.contrib[idx].contrib;

        if (path_host_.paths.attrib[idx].attr.is_terminated) {
            return false;
        }

        return true;
    }
}
