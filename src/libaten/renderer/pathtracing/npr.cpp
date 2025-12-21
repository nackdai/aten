#include <array>

#include "accelerator/threaded_bvh_traverser.h"
#include "material/material_impl.h"
#include "misc/omputil.h"
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
            Intersection isect;

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

                std::ignore = AT_NAME::HitShadowRay(
                    depth,
                    ctxt, mtrl,
                    path_host_.paths.attrib[idx],
                    path_host_.paths.contrib[idx],
                    shadow_rays_[idx]);

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
                    vec3 col = vec3(0);
                    vec3 col2 = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif
                    int32_t idx = y * width + x;
                    aten::hitrecord hrec;

                    for (uint32_t i = 0; i < samples; i++) {
                        const auto rnd = aten::getRandom(idx);
                        const auto& camsample = camera->param();

                        GeneratePath(
                            rays_[idx],
                            idx,
                            x, y, width, height,
                            i, GetFrameCount(),
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
                                ctxt, scene, camsample, &hrec);
                        }

                        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto c = path_host_.paths.contrib[idx].contrib;

                        col += c;
                        col2 += c * c;
                        cnt++;

                        if (path_host_.paths.attrib[idx].attr.is_terminated) {
                            break;
                        }
                    }

                    col /= (float)cnt;

#if 0
                    if (hrec.mtrlid >= 0) {
                        const auto mtrl = ctxt.GetMaterial(hrec.mtrlid);
                        if (mtrl && mtrl->isNPR()) {
                            col = FeatureLine::RenderFeatureLine(
                                col,
                                x, y, width, height,
                                hrec,
                                ctxt, *scene, *camera);
                        }
                    }
#endif

                    dst.buffer->put(x, y, vec4(col, 1));

                    if (dst.variance) {
                        col2 /= (float)cnt;
                        dst.variance->put(x, y, vec4(col2 - col * col, float(1)));
                    }
                }
            }
        }
    }
}
