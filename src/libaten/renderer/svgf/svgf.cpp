#include "renderer/svgf/svgf.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/svgf/svgf_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (10)
#define BREAK_Y    (511-447)
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

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].isHit = false;

            Intersection isect;
            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                path_host_.paths.attrib[idx].isHit = true;

                auto& aov = params_.GetCurrAovBuffer();
                auto aov_normal_depth = aov.GetNormalDepthAsSpan();
                auto aov_albedo_meshid = aov.GetAlbedoMeshIdAsSpan();
                Shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, russian_roulette_depth_, depth,
                    params_.mtxs.GetW2C(),
                    aov_normal_depth, aov_albedo_meshid);

                std::ignore = AT_NAME::HitShadowRay(
                    idx, depth, ctxt, path_host_.paths, shadow_rays_[idx]);

                willContinue = !path_host_.paths.attrib[idx].is_terminated;
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
                        bg_,
                        ctxt, camera,
                        path_host_.paths, rays_[idx],
                        aov.GetNormalDepthAsSpan(),
                        aov.GetAlbedoMeshIdAsSpan());
                }
                else {
                    ShadeMiss(
                        idx,
                        depth,
                        bg_.bg_color,
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
        const aten::mat4& mtx_W2C,
        aten::span<aten::vec4>& aov_normal_depth,
        aten::span<aten::vec4>& aov_albedo_meshid)
    {
        auto* sampler = &paths.sampler[idx];

        const auto& ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        // Render AOVs.
        // NOTE
        // 厳密に法線をAOVに保持するなら、法線マップ適用後するべき.
        // しかし、temporal reprojection、atrousなどのフィルタ適用時に法線を参照する際に、法線マップが細かすぎてはじかれてしまうことがある.
        // それにより、フィルタがおもったようにかからずフィルタの品質が下がってしまう問題が発生する.
        if (bounce == 0) {
            // texture color
            auto texcolor = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            AT_NAME::FillBasicAOVs(
                aov_normal_depth[idx], orienting_normal, rec, mtx_W2C,
                aov_albedo_meshid[idx], texcolor, isect);
            aov_albedo_meshid[idx].w = static_cast<float>(isect.mtrlid);

            // For exporting separated albedo.
            mtrl.albedoMap = -1;
        }
        // TODO
        // How to deal Refraction?
        else if (bounce == 1 && paths.attrib[idx].last_hit_mtrl_idx >= 0) {
            const auto& last_hit_mtrl = ctxt.GetMaterial(paths.attrib[idx].last_hit_mtrl_idx);
            if (last_hit_mtrl.type == aten::MaterialType::Specular) {
                // texture color.
                auto texcolor = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, aten::vec4(1.0f));

                // TODO
                // No good idea to compute reflected depth.
                AT_NAME::FillBasicAOVs(
                    aov_normal_depth[idx], orienting_normal, rec, mtx_W2C,
                    aov_albedo_meshid[idx], texcolor, isect);
                aov_albedo_meshid[idx].w = static_cast<float>(isect.mtrlid);

                // For exporting separated albedo.
                mtrl.albedoMap = -1;
            }
        }

        auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        auto& shadow_ray = shadow_rays[idx];
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

        // Check transparency or translucency.
        auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
            ctxt,
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

        aten::MaterialSampling sampling;
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

        aten::span contribs(reinterpret_cast<aten::vec4*>(paths.contrib), width * height);
        auto curr_aov_normal_depth{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto curr_aov_texclr_meshid{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto curr_aov_color_variance{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto curr_aov_moment_temporalweight{ curr_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };
        auto prev_aov_normal_depth{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto prev_aov_texclr_meshid{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto prev_aov_color_variance{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto prev_aov_moment_temporalweight{ prev_aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };

        const auto idx = ix + iy * width;

        auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel(
            idx,
            contribs,
            curr_aov_normal_depth,
            curr_aov_texclr_meshid);

        const auto center_meshid = aten::get<2>(extracted_center_pixel);
        auto curr_color{ aten::get<3>(extracted_center_pixel) };

        auto back_ground_pixel_clr = AT_NAME::svgf::UpdateAOVIfBackgroundPixel(
            idx, curr_color, center_meshid,
            curr_aov_color_variance, curr_aov_moment_temporalweight);
        if (back_ground_pixel_clr) {
            return back_ground_pixel_clr.value();
        }

        const auto center_normal{ aten::get<0>(extracted_center_pixel) };
        const float center_depth{ aten::get<1>(extracted_center_pixel) };

        float weight = 0;

        aten::tie(weight, curr_color) = AT_NAME::svgf::TemporalReprojection(
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

    std::optional<aten::vec4> SVGFRenderer::AtrousFilter(
        const int32_t filter_iter_count,
        const int32_t idx,
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const float camera_distance,
        AT_NAME::SVGFParams<std::vector<aten::vec4>>& svgf_param)
    {
        auto& curr_aov = svgf_param.GetCurrAovBuffer();

        auto atrous_clr_variance_buffers = svgf_param.GetAtrousColorVariance(filter_iter_count);
        auto& curr_atrous_clr_variance = aten::get<0>(atrous_clr_variance_buffers);
        auto& next_atrous_clr_variance = aten::get<1>(atrous_clr_variance_buffers);

        bool isFirstIter = (filter_iter_count == 0);
        bool isFinalIter = (filter_iter_count == svgf_param.atrous_iter_cnt - 1);

        aten::span temporary_color_buffer(svgf_param.temporary_color_buffer);
        auto aov_normal_depth{ curr_aov.GetAsConstSpan<AT_NAME::SVGFAovBufferType::NormalDepth>() };
        auto aov_texclr_meshid{ curr_aov.GetAsConstSpan<AT_NAME::SVGFAovBufferType::AlbedoMeshId>() };
        auto aov_color_variance{ curr_aov.GetAsConstSpan<AT_NAME::SVGFAovBufferType::ColorVariance>() };
        auto aov_moment_temporalweight{ curr_aov.GetAsConstSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>() };
        aten::const_span color_variance_buffer(curr_atrous_clr_variance);
        aten::span next_color_variance_buffer(next_atrous_clr_variance);

        auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel<false>(
            idx,
            isFirstIter ? aov_color_variance : color_variance_buffer,
            aov_normal_depth,
            aov_texclr_meshid);

        const auto center_normal{ aten::get<0>(extracted_center_pixel) };
        const float center_depth{ aten::get<1>(extracted_center_pixel) };
        const auto center_meshid{ aten::get<2>(extracted_center_pixel) };
        auto center_color{ aten::get<3>(extracted_center_pixel) };

        auto back_ground_pixel_clr = AT_NAME::svgf::CheckIfBackgroundPixelForAtrous(
            isFinalIter, idx,
            center_color,
            aov_texclr_meshid, next_color_variance_buffer);

        if (back_ground_pixel_clr) {
            if (back_ground_pixel_clr.value()) {
                // Output background color and end the logic.
                const auto& background = back_ground_pixel_clr.value().value();
                return background;
            }
        }

        // 3x3 Gauss filter.
        auto gauss_filtered_variance = AT_NAME::svgf::Exec3x3GaussFilter(
            ix, iy, width, height,
            isFirstIter ? aov_color_variance : color_variance_buffer,
            &aten::vec4::w);

        auto filtered_color_variance{
            AT_NAME::svgf::ExecAtrousWaveletFilter(
                isFirstIter,
                ix, iy, width, height,
                gauss_filtered_variance,
                center_normal, center_depth, center_meshid, center_color,
                aov_normal_depth, aov_texclr_meshid, aov_color_variance,
                color_variance_buffer,
                filter_iter_count, camera_distance)
        };

        auto post_process_result = AT_NAME::svgf::PostProcessForAtrousFilter(
            isFirstIter, isFinalIter,
            idx,
            filtered_color_variance,
            aov_texclr_meshid,
            temporary_color_buffer, next_color_variance_buffer);

        if (post_process_result) {
            return post_process_result.value();
        }

        return std::nullopt;
    }

    void SVGFRenderer::CopyFromTeporaryColorBufferToAov(
        const int32_t idx,
        AT_NAME::SVGFParams<std::vector<aten::vec4>>& svgf_param)
    {
        auto& curaov = svgf_param.GetCurrAovBuffer();

        auto& aov_clr_variance = curaov.get<AT_NAME::SVGFAovBufferType::ColorVariance>();

        aten::const_span src(svgf_param.temporary_color_buffer);
        aten::span<aten::vec4> dst(aov_clr_variance);

        AT_NAME::svgf::CopyVectorBuffer<3>(idx, src, dst);
    }

    void SVGFRenderer::Initialize(
        const Destination& dst,
        const Camera& camera)
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
        path_host_.Clear(GetFrameCount());

        params_.InitBuffers(width, height);
        params_.mtxs.Reset(camera.param());
    }

    void SVGFRenderer::SetMotionDepthBuffer(aten::FBO& fbo, int32_t idx)
    {
        const auto width = fbo.GetWidth();
        const auto height = fbo.GetHeight();
        params_.motion_depth_buffer.resize(width * height);

        aten::span motion_depth_buffer(params_.motion_depth_buffer);
        fbo.SaveToBuffer(motion_depth_buffer, 1);
    }

    void SVGFRenderer::OnRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
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
#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif
                    int32_t idx = y * width + x;

                    for (uint32_t i = 0; i < samples; i++) {
                        const auto rnd = aten::getRandom(idx);
                        const auto& camsample = camera->param();

                        GeneratePath(
                            rays_[idx],
                            idx,
                            x, y,
                            i, GetFrameCount(),
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

                        if (path_host_.paths.attrib[idx].is_terminated) {
                            break;
                        }
                    }

                    auto& aov = params_.GetCurrAovBuffer();

                    aten::span tmp_color_buffer(params_.temporary_color_buffer);

                    if (GetFrameCount() == 0) {
                        AT_NAME::svgf::PrepareForDenoise<true>(
                            idx,
                            path_host_.paths,
                            tmp_color_buffer,
                            aov.GetAsSpan<AT_NAME::SVGFAovBufferType::ColorVariance>(),
                            aov.GetAsSpan<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>());
                    }
                    else {
                        AT_NAME::svgf::PrepareForDenoise<false>(
                            idx,
                            path_host_.paths,
                            tmp_color_buffer);
                    }

                    dst.buffer->put(x, y, path_host_.paths.contrib[idx].contrib);
                }
            }

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

                    aten::vec4 temporal_projected_clr;
                    if (GetFrameCount() > 0) {
                        temporal_projected_clr = TemporalReprojection(
                            x, y, width, height,
                            0.98f, 0.05f,
                            path_host_.paths,
                            camera->param(),
                            params_);
                    }
                    else {
                        temporal_projected_clr = path_host_.paths.contrib[idx].contrib;
                    }

                    dst.buffer->put(x, y, temporal_projected_clr);
                }
            }

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

                    auto camera_distance = AT_NAME::Camera::ComputeScreenDistance(camera->param(), height);
                    auto variance = EstimateVariance(
                        x, y, width, height,
                        camera_distance,
                        params_);

                    dst.buffer->put(x, y, variance);
                }
            }

            auto camera_distance = AT_NAME::Camera::ComputeScreenDistance(camera->param(), height);

            for (int32_t i = 0; i < params_.atrous_iter_cnt; i++) {
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

                        std::optional<aten::vec4> filtered_color;

                        filtered_color = AtrousFilter(
                            i,
                            idx, x, y, width, height,
                            camera_distance,
                            params_);
                        if (filtered_color) {
                            dst.buffer->put(x, y, filtered_color.value());
                        }
                    }
                }
            }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    int32_t idx = y * width + x;
                    CopyFromTeporaryColorBufferToAov(idx, params_);
                }
            }
        }

        // Toggle aov buffer pos.
        params_.UpdateCurrAovBufferPos();
    }
}
