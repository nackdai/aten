#include "renderer/restir/restir.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing.h"
#include "renderer/restir/restir_impl.h"

#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void ReSTIRRenderer::Render(
        int32_t idx,
        int32_t x, int32_t y,
        int32_t sample_cnt,
        int32_t bounce,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera)
    {
        const auto rnd = aten::getRandom(idx);

        GeneratePath(
            rays_[idx],
            idx,
            x, y,
            sample_cnt, GetFrameCount(),
            path_host_.paths,
            camera,
            rnd);

        const auto& ray = rays_[idx];

        path_host_.paths.attrib[idx].isHit = false;

        Intersection isect;
        if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
            path_host_.paths.attrib[idx].isHit = true;

            if (bounce == 0) {
                aten::span reservoirs(reservoirs_.GetCurrParams());
                aten::span restir_infos(restir_infos_.GetCurrParams());

                auto aov_normal_depth = aov_.GetNormalDepthAsSpan();
                auto aov_albedo_meshid = aov_.GetAlbedoMeshIdAsSpan();

                Shade(
                    idx,
                    width, height,
                    path_host_.paths, ctxt, rays_.data(), isect,
                    reservoirs, restir_infos,
                    russian_roulette_depth_, bounce,
                    mtxs_.GetW2C(),
                    aov_normal_depth, aov_albedo_meshid);
            }
            else {
                PathTracing::shade(
                    idx,
                    path_host_.paths, ctxt,
                    rays_.data(), shadow_rays_.data(),
                    isect, scene,
                    russian_roulette_depth_, bounce);

                std::ignore = AT_NAME::HitShadowRay(
                    idx, bounce, ctxt, path_host_.paths, shadow_rays_[idx], scene);
            }
        }
        else {
            auto ibl = scene->getIBL();
            if (ibl) {
                ShadeMissWithEnvmap(
                    idx,
                    x, y,
                    width, height,
                    bounce,
                    bg_,
                    ctxt, camera,
                    path_host_.paths, rays_[idx],
                    aov_.GetNormalDepthAsSpan(),
                    aov_.GetAlbedoMeshIdAsSpan());
            }
            else {
                ShadeMiss(
                    idx,
                    bounce,
                    bg_.bg_color,
                    path_host_.paths,
                    aov_.GetNormalDepthAsSpan(),
                    aov_.GetAlbedoMeshIdAsSpan());
            }
        }
    }

    void ReSTIRRenderer::Shade(
        int32_t idx,
        int32_t width, int32_t height,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        const aten::Intersection& isect,
        aten::span<AT_NAME::Reservoir>& reservoirs,
        aten::span<AT_NAME::ReSTIRInfo>& restir_infos,
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

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        // Apply normal map.
        int32_t normalMap = mtrl.normalMap;
        const auto pre_sampled_r = AT_NAME::material::applyNormal(
            &mtrl,
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths.sampler[idx]);

        if (!mtrl.attrib.isTranslucent
            && !mtrl.attrib.isEmissive
            && isBackfacing)
        {
            orienting_normal = -orienting_normal;
        }

        auto& restir_info = restir_infos[idx];
        {
            restir_info.clear();
            restir_info.nml = orienting_normal;
            restir_info.is_voxel = rec.isVoxel;
            restir_info.mtrl_idx = rec.mtrlid;
            restir_info.wi = ray.dir;
            restir_info.u = rec.u;
            restir_info.v = rec.v;
            restir_info.p = rec.p;
            restir_info.pre_sampled_r = pre_sampled_r;
            restir_info.mesh_id = isect.meshid;
        }

        if (bounce == 0) {
            // Store AOV.
            // World coordinate to Clip coordinate.
            aten::vec4 pos = aten::vec4(rec.p, 1);
            pos = mtx_W2C.apply(pos);

            aov_normal_depth[idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);
            aov_albedo_meshid[idx] = make_float4(albedo.x, albedo.y, albedo.z, static_cast<float>(isect.meshid));
        }

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

        // Generate initial candidates.
        if (!(mtrl.attrib.isSingular || mtrl.attrib.isTranslucent))
        {
            auto& reservoir = reservoirs[idx];

            AT_NAME::restir::GenerateInitialCandidate<std::remove_cv_t<decltype(ctxt)>, 1>(
                reservoir,
                mtrl,
                ctxt,
                rec.p, orienting_normal,
                ray.dir,
                rec.u, rec.v,
                &paths.sampler[idx],
                pre_sampled_r,
                bounce);
        }

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

        AT_NAME::PrepareForNextBounce(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            mtrl, sampling, albedo,
            paths, rays);
    }

    void ReSTIRRenderer::EvaluateVisibility(
        int32_t idx,
        int32_t bounce,
        int32_t width, int32_t height,
        aten::Path& paths,
        const context& ctxt,
        aten::scene* scene,
        std::vector<AT_NAME::Reservoir>& reservoirs,
        std::vector<AT_NAME::ReSTIRInfo>& restir_infos,
        std::vector<AT_NAME::ShadowRay>& shadowRays)
    {
        aten::span shadow_rays(shadowRays);

        AT_NAME::restir::EvaluateVisibility(
            idx,
            bounce,
            paths,
            ctxt,
            reservoirs[idx],
            restir_infos[idx],
            shadow_rays,
            scene);
    }

    void ReSTIRRenderer::ApplySpatialReuse(
        int32_t idx,
        int32_t width, int32_t height,
        const aten::context& ctxt,
        aten::sampler& sampler,
        const std::vector<AT_NAME::Reservoir>& curr_reservoirs,
        std::vector<AT_NAME::Reservoir>& dst_reservoirs,
        const std::vector<AT_NAME::ReSTIRInfo>& infos,
        const std::vector<aten::vec4>& aovTexclrMeshid)
    {
        aten::const_span reservoirs_as_span(curr_reservoirs);
        aten::const_span resitr_infos(infos);
        aten::const_span aov_albedo_meshid(aovTexclrMeshid);

        AT_NAME::restir::ApplySpatialReuse(
            idx,
            width, height,
            ctxt,
            sampler,
            dst_reservoirs[idx],
            reservoirs_as_span,
            resitr_infos,
            aov_albedo_meshid);
    }

    void ReSTIRRenderer::ApplyTemporalReuse(
        int32_t idx,
        int32_t width, int32_t height,
        const aten::context& ctxt,
        aten::sampler& sampler,
        std::vector<AT_NAME::Reservoir>& reservoir,
        const std::vector<AT_NAME::Reservoir>& prev_reservoirs,
        const std::vector<AT_NAME::ReSTIRInfo>& curr_infos,
        const std::vector<AT_NAME::ReSTIRInfo>& prev_infos,
        const std::vector<aten::vec4>& aovTexclrMeshid,
        const std::vector<aten::vec4>& motion_depth_buffer)
    {
        aten::const_span prev_reservoirs_as_span(prev_reservoirs);
        aten::const_span prev_resitr_infos(prev_infos);
        aten::const_span aov_albedo_meshid(aovTexclrMeshid);

        AT_NAME::restir::ApplyTemporalReuse(
            idx,
            width, height,
            ctxt,
            sampler,
            reservoir[idx],
            curr_infos[idx],
            prev_reservoirs_as_span,
            prev_resitr_infos,
            aov_albedo_meshid,
            motion_depth_buffer);
    }

    void ReSTIRRenderer::Initialize(
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
        path_host_.Clear(GetFrameCount());

        reservoirs_.Init(width * height);
        restir_infos_.Init(width * height);

        mtxs_.Reset(camera.param());

        aov_.traverse(
            [width, height](std::vector<vec4>& buffer)
            {
                if (buffer.empty()) {
                    buffer.resize(width * height);
                }
            });

        for (auto& attrib : path_host_.attrib) {
            attrib.isKill = false;
        }
    }

    void ReSTIRRenderer::ComputePixelColor(
        int32_t idx,
        AT_NAME::Path& paths,
        const aten::context& ctxt,
        const std::vector<AT_NAME::Reservoir>& reservoirs,
        const std::vector<AT_NAME::ReSTIRInfo>& restir_infos,
        const std::vector<aten::vec4>& aov_albedo_meshid)
    {
        const auto& reservoir = reservoirs[idx];
        const auto& restir_info = restir_infos[idx];

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            restir_info.mtrl_idx,
            restir_info.is_voxel);

        auto pixel_color = AT_NAME::restir::ComputePixelColor(
            ctxt,
            reservoir, restir_info,
            mtrl,
            aov_albedo_meshid[idx]);
        if (pixel_color) {
            const auto result = pixel_color.value() * paths.throughput[idx].throughput;
            paths.contrib[idx].contrib += make_float3(result.x, result.y, result.z);
        }
    }

    void ReSTIRRenderer::SetMotionDepthBuffer(aten::FBO& fbo, int32_t idx)
    {
        const auto width = fbo.GetWidth();
        const auto height = fbo.GetHeight();
        motion_depth_buffer_.resize(width * height);

        aten::span motion_depth_buffer(motion_depth_buffer_);
        fbo.SaveToBuffer(motion_depth_buffer, 1);
    }

    void ReSTIRRenderer::OnRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        const auto& cam_param = camera->param();

        Initialize(dst, *camera);

        auto time = timer::getSystemTime();

        for (uint32_t i = 0; i < samples; i++) {
            int32_t bounce = 0;

            while (bounce < max_depth_) {
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
                            int32_t idx = y * width + x;

                            if (path_host_.paths.attrib[idx].isKill || path_host_.paths.attrib[idx].isTerminate) {
                                continue;
                            }

                            Render(
                                idx, x, y,
                                i, bounce,
                                width, height,
                                ctxt, scene, cam_param);

                            dst.buffer->put(x, y, path_host_.paths.contrib[idx].contrib);
                        }
                    }

                    if (bounce > 0) {
                        bounce++;
                        continue;
                    }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                    for (int32_t y = 0; y < height; y++) {
                        for (int32_t x = 0; x < width; x++) {
                            int32_t idx = y * width + x;

                            if (path_host_.paths.attrib[idx].isTerminate) {
                                continue;
                            }

                            EvaluateVisibility(
                                idx,
                                bounce,
                                width, height,
                                path_host_.paths, ctxt, scene,
                                reservoirs_.GetCurrParams(),
                                restir_infos_.GetCurrParams(),
                                shadow_rays_);
                        }
                    }

                    const auto curr_reservoirs_idx = reservoirs_.GetCurrParamsIdx();
                    const auto curr_restir_infos_idx = restir_infos_.GetCurrParamsIdx();

                    auto target_reservoirs_idx = curr_reservoirs_idx;
                    auto target_restir_infos_idx = curr_restir_infos_idx;

                    if (bounce == 0) {
                        const auto dst_reservoirs_idx = reservoirs_.GetDestinationParamsIdxForSpatialReuse();

                        auto& curr_reservoirs = reservoirs_.GetCurrParams();
                        const auto& prev_reservoirs = reservoirs_.GetPreviousFrameParamsForTemporalReuse();
                        auto& dst_reservoirs = reservoirs_.GetDestinationParamsForSpatialReuse();

                        const auto& curr_restir_infos = restir_infos_.GetCurrParams();
                        const auto& prev_restir_infos = restir_infos_.GetPreviousFrameParamsForTemporalReuse();

                        target_reservoirs_idx = dst_reservoirs_idx;

                        reservoirs_.Update();
                        restir_infos_.Update();

#if 1
                        const auto frame = GetFrameCount();
                        if (frame > 1) {
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                            for (int32_t y = 0; y < height; y++) {
                                for (int32_t x = 0; x < width; x++) {
                                    int32_t idx = y * width + x;

                                    if (path_host_.paths.attrib[idx].isTerminate) {
                                        continue;
                                    }

                                    ApplyTemporalReuse(
                                        idx,
                                        width, height,
                                        ctxt,
                                        path_host_.sampler[idx],
                                        curr_reservoirs,
                                        prev_reservoirs,
                                        curr_restir_infos, prev_restir_infos,
                                        aov_.albedo_meshid(),
                                        motion_depth_buffer_);
                                }
                            }
                        }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                        for (int32_t y = 0; y < height; y++) {
                            for (int32_t x = 0; x < width; x++) {
                                int32_t idx = y * width + x;

                                if (path_host_.paths.attrib[idx].isTerminate) {
                                    continue;
                                }

                                ApplySpatialReuse(
                                    idx,
                                    width, height,
                                    ctxt,
                                    path_host_.sampler[idx],
                                    curr_reservoirs,
                                    dst_reservoirs,
                                    curr_restir_infos,
                                    aov_.albedo_meshid());
                            }
                        }
#endif
                    }

#if 1
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
                    for (int32_t y = 0; y < height; y++) {
                        for (int32_t x = 0; x < width; x++) {
                            int32_t idx = y * width + x;

                            if (path_host_.paths.attrib[idx].isTerminate) {
                                continue;
                            }

                            ComputePixelColor(
                                idx,
                                path_host_.paths,
                                ctxt,
                                reservoirs_.GetParams(target_reservoirs_idx),
                                restir_infos_.GetParams(target_restir_infos_idx),
                                aov_.albedo_meshid());

                            dst.buffer->put(x, y, path_host_.paths.contrib[idx].contrib);
                        }
                    }
#endif

                    bounce++;
                }
            }
        }
    }
}
