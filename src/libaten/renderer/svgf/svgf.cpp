#include "renderer/svgf/svgf.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/svgf/svgf_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void SVGFRenderer::ExecRendering(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera)
    {
        int32_t depth = 0;

        while (depth < max_depth_) {
            bool willContinue = true;
            Intersection isect;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].isHit = false;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                path_host_.paths.attrib[idx].isHit = true;

                auto& aov = params_.GetCurrAovBuffer();
                Shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, russian_roulette_depth_, depth,
                    params_.mtxs,
                    aov.GetNormalDepthAsSpan(),
                    aov.GetAlbedoMeshIdAsSpan());

                std::ignore = AT_NAME::HitShadowRay(
                    idx, depth, ctxt, path_host_.paths, shadow_rays_.data(), scene);

                willContinue = !path_host_.paths.attrib[idx].isTerminate;
            }
            else {
                auto ibl = scene->getIBL();
                auto& aov = params_.GetCurrAovBuffer();
                if (ibl) {
                    ShadeMissWithEnvmap(
                        idx,
                        ix, iy,
                        width, height,
                        depth,
                        ibl->param().GetLightEnvmapId(),
                        ibl->getAvgIlluminace(),
                        real(1),
                        ctxt, camera,
                        path_host_.paths, rays_[idx],
                        aov.GetNormalDepthAsSpan(),
                        aov.GetAlbedoMeshIdAsSpan());
                }
                else {
                    ShadeMiss(
                        idx,
                        depth,
                        bg()->sample(rays_[idx]),
                        path_host_.paths,
                        aov.GetNormalDepthAsSpan(),
                        aov.GetAlbedoMeshIdAsSpan());
                }

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }

    void SVGFRenderer::Shade(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        aten::ShadowRay* shadow_rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce,
        const AT_NAME::SVGFMtxPack mtxs,
        aten::span<aten::vec4>& aov_normal_depth,
        aten::span<aten::vec4>& aov_albedo_meshid)
    {
        auto* sampler = &paths.sampler[idx];

        const auto& ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < real(0);

        // �����ʒu�̖@��.
        // ���̂���̃��C�̓��o���l��.
        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        // Implicit conection to light.
        auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
            isBackfacing,
            bounce,
            paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
            ray,
            rec.p, orienting_normal,
            rec.area,
            mtrl);
        if (is_hit_implicit_light) {
            return;
        }

        AT_NAME::svgf::FillAOVs<true, true, true>(
            idx, bounce,
            paths, rec, isect,
            mtxs.GetW2C(),
            orienting_normal, mtrl,
            aov_normal_depth, aov_albedo_meshid);

        if (!mtrl.attrib.isTranslucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        auto pre_sampled_r = material::applyNormal(
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, sampler);

        // Check transparency or translucency.
        auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
            mtrl,
            rec.u, rec.v, rec.p,
            orienting_normal,
            rays[idx],
            paths.sampler[idx],
            paths.attrib[idx],
            paths.throughput[idx]);
        if (is_translucent_by_alpha) {
            return;
        }

        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            shadow_ray,
            ctxt,
            bounce,
            paths.sampler[idx],
            paths.throughput[idx],
            mtrl,
            ray,
            rec.p, orienting_normal,
            rec.u, rec.v, albedo,
            pre_sampled_r);

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);

        aten::MaterialSampling sampling;
        material::sampleMaterial(
            &sampling,
            &mtrl,
            orienting_normal,
            ray.dir,
            rec.normal,
            sampler, pre_sampled_r,
            rec.u, rec.v);

        AT_NAME::PostProcessPathTrancing(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            mtrl, sampling,
            paths, rays);
    }

    aten::vec4 SVGFRenderer::TemporalReprojection(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const float threshold_normal,
        const float threshold_depth,
        const AT_NAME::Path& paths,
        const aten::CameraParameter& camera,
        AT_NAME::SVGFParams<std::vector<aten::vec4>>& svgf_param)
    {
        auto& curr_aov = svgf_param.GetCurrAovBuffer();
        auto& prev_aov = svgf_param.GetPrevAovBuffer();

        aten::span<aten::vec4> contribs(reinterpret_cast<aten::vec4*>(paths.contrib), width * height);
        auto curr_aov_normal_depth{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto curr_aov_texclr_meshid{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto curr_aov_color_variance{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto curr_aov_moment_temporalweight{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };
        auto prev_aov_normal_depth{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto prev_aov_texclr_meshid{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto prev_aov_color_variance{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto prev_aov_moment_temporalweight{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };

        const auto idx = ix * iy * width;

        auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel(
            idx,
            contribs,
            curr_aov_normal_depth,
            curr_aov_texclr_meshid);

        const auto center_meshid = aten::get<2>(extracted_center_pixel);
        auto curr_color = aten::get<3>(extracted_center_pixel);

        auto back_ground_pixel_clr = AT_NAME::svgf::CheckIfBackgroundPixel(
            idx, curr_color, center_meshid,
            curr_aov_color_variance, curr_aov_moment_temporalweight);
        if (back_ground_pixel_clr) {
            return back_ground_pixel_clr.value();
        }

        const auto center_normal = aten::get<0>(extracted_center_pixel);
        const float center_depth = aten::get<1>(extracted_center_pixel);

        auto weight = AT_NAME::svgf::TemporalReprojection(
            ix, iy, width, height,
            threshold_normal, threshold_depth,
            center_normal, center_depth, center_meshid,
            curr_color,
            curr_aov_color_variance, curr_aov_moment_temporalweight,
            prev_aov_normal_depth, prev_aov_texclr_meshid, prev_aov_color_variance,
            svgf_param.motion_depth_buffer);

        AT_NAME::svgf::AccumulateMoments(
            idx, weight,
            curr_aov_color_variance,
            curr_aov_moment_temporalweight,
            prev_aov_moment_temporalweight);

        return curr_color;
    }

    aten::vec4 SVGFRenderer::EstimateVariance(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const float camera_distance,
        AT_NAME::SVGFParams<std::vector<aten::vec4>>& svgf_param)
    {
        auto& curr_aov = svgf_param.GetCurrAovBuffer();

        auto aov_normal_depth{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto aov_texclr_meshid{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto aov_color_variance{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto aov_moment_temporalweight{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };

        return AT_NAME::svgf::EstimateVariance(
            ix, iy, width, height,
            svgf_param.mtxs.mtx_C2V,
            camera_distance,
            aov_normal_depth,
            aov_texclr_meshid,
            aov_color_variance,
            aov_moment_temporalweight);
    }

    void SVGFRenderer::Initialize(
        const Destination& dst,
        const camera& camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        max_depth_ = dst.maxDepth;
        russian_roulette_depth_ = dst.russianRouletteDepth;

        if (russian_roulette_depth_ > max_depth_) {
            russian_roulette_depth_ = max_depth_ - 1;
        }

        if (rays_.empty()) {
            rays_.resize(width * height);
        }
        if (shadow_rays_.empty()) {
            shadow_rays_.resize(width * height);
        }
        path_host_.init(width, height);

        params_.InitBuffers(width, height);
        params_.mtxs.Reset(camera.param());

        for (auto& attrib : path_host_.attrib) {
            attrib.isKill = false;
        }
    }

    void SVGFRenderer::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        Initialize(dst, *camera);

        auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto thread_idx = OMPUtil::getThreadIdx();

            auto t = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    vec3 col = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif
                    int32_t idx = y * width + x;

                    if (path_host_.paths.attrib[idx].isKill) {
                        continue;
                    }

                    for (uint32_t i = 0; i < samples; i++) {
                        const auto rnd = aten::getRandom(idx);
                        const auto& camsample = camera->param();

                        GeneratePath(
                            rays_[idx],
                            idx,
                            x, y,
                            i, get_frame_count(),
                            path_host_.paths,
                            camsample,
                            rnd);

                        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

                        ExecRendering(
                            idx,
                            x, y, width, height,
                            ctxt, scene, camera->param());

                        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto& aov = params_.GetCurrAovBuffer();

                        if (get_frame_count() == 0) {
                            AT_NAME::svgf::PrepareForDenoise<true>(
                                idx,
                                path_host_.paths,
                                aten::span<decltype(params_)::buffer_value_type>(params_.temporary_color_buffer),
                                aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>(),
                                aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>());
                        }
                        else {
                            AT_NAME::svgf::PrepareForDenoise<false>(
                                idx,
                                path_host_.paths,
                                aten::span<decltype(params_)::buffer_value_type>(params_.temporary_color_buffer));
                        }

                        auto teporal_projected_clr = TemporalReprojection(
                            x, y, width, height,
                            0.98f, 0.05f,
                            path_host_.paths,
                            camera->param(),
                            params_);

                        auto camera_distance = AT_NAME::camera::ComputeScreenDistance(camera->param(), height);
                        auto variance = EstimateVariance(
                            x, y, width, height,
                            camera_distance,
                            params_);

                        auto c = path_host_.paths.contrib[idx].contrib;

                        col += c;
                        cnt++;

                        if (path_host_.paths.attrib[idx].isTerminate) {
                            break;
                        }
                    }

                    col /= (real)cnt;

                    dst.buffer->put(x, y, vec4(col, 1));
                }
            }
        }
    }
}