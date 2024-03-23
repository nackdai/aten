#pragma once

#include <optional>

#include "light/light.h"
#include "material/material.h"
#include "math/mat4.h"
#include "misc/span.h"
#include "misc/const_span.h"
#include "misc/tuple.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/pathtracing/pt_params.h"

#include "restir_types.h"

namespace AT_NAME {
namespace restir {
    namespace _detail {
        inline AT_DEVICE_API auto ComputeRadiance(
            const aten::LightSampleResult& lightsample,
            const aten::LightAttribute& light_attrib,
            const aten::vec3& normal,
            const aten::vec3& ray_dir,
            const aten::MaterialParameter& mtrl,
            const float u, const float v,
            const aten::vec4& albedo,
            real pre_sampled_r)
        {
            aten::vec3 nmlLight = lightsample.nml;
            aten::vec3 dirToLight = normalize(lightsample.dir);

            const auto cosShadow = aten::abs(dot(normal, dirToLight));
            const auto cosLight = aten::abs(dot(nmlLight, -dirToLight));
            const auto dist2 = aten::squared_length(lightsample.dir);

            auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                &mtrl, normal,
                ray_dir, dirToLight,
                u, v,
                albedo,
                pre_sampled_r);

            const auto geometry_term = light_attrib.isSingular || light_attrib.isInfinite
                ? cosShadow * cosLight
                : cosShadow * cosLight / dist2;

            auto le = brdf * lightsample.light_color * geometry_term;
            return le;
        }

        inline AT_DEVICE_API float ComputeTargetPDF(
            const aten::LightSampleResult& lightsample,
            const aten::LightAttribute& light_attrib,
            const aten::vec3& normal,
            const aten::vec3& ray_dir,
            const aten::MaterialParameter& mtrl,
            const float u, const float v,
            real pre_sampled_r)
        {
            aten::vec3 nmlLight = lightsample.nml;
            aten::vec3 dirToLight = normalize(lightsample.dir);

            auto pdf = AT_NAME::material::samplePDF(&mtrl, normal, ray_dir, dirToLight, u, v);
            if (pdf == 0.0f) {
                return 0.0f;
            }

            // NOTE:
            // Apply albedo at the final phase to compute pixel color.
            // So, specify albedo color (1, 1, 1) temporarily.
            auto energy = ComputeRadiance(
                lightsample, light_attrib,
                normal, ray_dir, mtrl,
                u, v,
                aten::vec4(1.0f),
                pre_sampled_r);
            energy /= pdf;

            auto target_pdf = (energy.x + energy.y + energy.z) / 3;

            return target_pdf;
        }
    }

    template<class CONTEXT>
    inline AT_DEVICE_API int32_t SampleLightByStreamingRIS(
        AT_NAME::Reservoir& reservoir,
        const aten::MaterialParameter& mtrl,
        const CONTEXT& ctxt,
        const aten::vec3& org,
        const aten::vec3& normal,
        const aten::vec3& ray_dir,
        float u, float v,
        aten::sampler* sampler,
        real pre_sampled_r,
        int32_t lod = 0)
    {
        constexpr auto MaxLightCount = 32U;

        const auto light_num = ctxt.GetLightNum();
        const auto max_light_num = static_cast<decltype(MaxLightCount)>(light_num);
        const auto light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

        reservoir.clear();

        real candidate_target_pdf = real(0);

        real light_select_prob = real(1) / max_light_num;

        for (auto i = 0U; i < light_cnt; i++) {
            const auto r_light = sampler->nextSample();
            const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

            const auto& light = ctxt.GetLight(light_pos);

            aten::LightSampleResult lightsample;
            AT_NAME::Light::sample(lightsample, light, ctxt, org, normal, sampler, lod);

            aten::vec3 nmlLight = lightsample.nml;
            aten::vec3 dirToLight = normalize(lightsample.dir);

            const auto cosShadow = aten::abs(dot(normal, dirToLight));
            const auto cosLight = aten::abs(dot(nmlLight, -dirToLight));
            const auto dist2 = aten::squared_length(lightsample.dir);

            // NOTE:
            // Regarding punctual light, nothing to sample.
            // It means there is nothing to convert pdf.
            // TODO: IBL...
            auto path_pdf = lightsample.pdf;
            if (!light.attrib.isSingular && !light.attrib.isIBL) {
                // Convert solid angle PDF to area PDF.
                path_pdf = path_pdf * cosLight / dist2;
            }

            // p
            auto sampling_pdf = path_pdf * light_select_prob;

            // p_hat
            auto target_pdf = _detail::ComputeTargetPDF(
                lightsample, light.attrib,
                normal, ray_dir,
                mtrl,
                u, v,
                pre_sampled_r);

            // NOTE
            // Equation(5)
            // w(x) = p_hat(x) / p(x)
            auto weight = sampling_pdf > 0
                ? target_pdf / sampling_pdf
                : 0.0f;

            auto r = sampler->nextSample();

            if (reservoir.update(lightsample, light_pos, weight, r)) {
                candidate_target_pdf = target_pdf;
            }
        }

        if (candidate_target_pdf > 0.0f) {
            reservoir.target_pdf_of_y = candidate_target_pdf;
            // NOTE:
            // Equation(6)
            // W = 1/p_hat(x) * (1/M * w_sum) = w_sum / (p_hat(x) * M)
            reservoir.W = reservoir.w_sum / (reservoir.target_pdf_of_y * reservoir.M);
        }

        if (!isfinite(reservoir.W)) {
            reservoir.clear();
        }

        return reservoir.y;
    }

    template <class CONTEXT>
    inline AT_DEVICE_API void EvaluateVisibility(
        int32_t idx,
        int32_t bounce,
        AT_NAME::Path& paths,
        CONTEXT& ctxt,
        aten::span<AT_NAME::Reservoir>& reservoirs,
        const aten::const_span<AT_NAME::ReSTIRInfo>& restir_infos,
        aten::span<AT_NAME::ShadowRay>& shadowRays)
    {
        bool isHit = false;

        const auto& reservoir = reservoirs[idx];

        if (reservoir.IsValid()) {
            const auto& restir_info = restir_infos[idx];

            shadowRays[idx].rayorg = restir_info.p + AT_MATH_EPSILON * restir_info.nml;
            shadowRays[idx].raydir = reservoir.light_sample_.pos - shadowRays[idx].rayorg;
            shadowRays[idx].targetLightId = reservoir.y;
            shadowRays[idx].isActive = true;

            auto dist = length(shadowRays[idx].raydir);;
            shadowRays[idx].distToLight = dist;
            shadowRays[idx].raydir /= dist;

            isHit = AT_NAME::HitShadowRay(idx, bounce, ctxt, paths, shadowRays[idx]);
        }

        if (!isHit) {
            // NOTE:
            // In the function to combine the streams of multiple reservoirs (Alg.4), M is used to compute sum of samples.
            // So, M should not be cleared.
            reservoirs[idx].w_sum = 0.0f;
            reservoirs[idx].W = 0.0f;
            reservoirs[idx].target_pdf_of_y = 0.0f;
            reservoirs[idx].y = -1;
        }
    }

    namespace _detail {
        inline AT_DEVICE_API bool IsAcceptableNeighbor(
            const aten::MaterialParameter& mtrl,
            const int32_t mesh_id,
            const aten::vec3& normal,
            const aten::MaterialParameter& neightbor_mtrl,
            const int32_t neighbor_mesh_id,
            const aten::vec3& neighbor_normal)
        {
            constexpr auto NormalThreshold = 0.95f;

            return (mtrl.type == neightbor_mtrl.type)
                && (mesh_id == neighbor_mesh_id)
                && (dot(normal, neighbor_normal) >= NormalThreshold);
        }
    }

    template <class CONTEXT, class BufferForMotionDepth>
    inline AT_DEVICE_API void ApplyTemporalReuse(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        CONTEXT& ctxt,
        aten::sampler& sampler,
        AT_NAME::Reservoir& combined_reservoir,
        const AT_NAME::ReSTIRInfo& self_info,
        const aten::const_span<AT_NAME::Reservoir>& prev_reservoirs,
        const aten::const_span<AT_NAME::ReSTIRInfo>& infos,
        const aten::const_span<AT_NAME::_detail::v4>& aov_albedo_meshid,
        BufferForMotionDepth& motion_detph_buffer)
    {
        const auto idx = getIdx(ix, iy, width);

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            self_info.mtrl_idx,
            self_info.is_voxel);

        const auto& normal = self_info.nml;

        const auto mesh_id = static_cast<int32_t>(aov_albedo_meshid[idx].w);

        float candidate_target_pdf = combined_reservoir.IsValid()
            ? combined_reservoir.target_pdf_of_y
            : 0.0f;

        // NOTE
        // In this case, self reservoir's M should be number of light sampling.
        const auto maxM = 20 * combined_reservoir.M;

        AT_NAME::_detail::v4 motion_depth;
        if constexpr (std::is_class_v<std::remove_reference_t<decltype(motion_detph_buffer)>>) {
            motion_depth = motion_detph_buffer[idx];
        }
        else {
            surf2Dread(&motion_depth, motion_detph_buffer, ix * sizeof(motion_depth), iy);
        }

        // 前のフレームのスクリーン座標.
        int32_t px = (int32_t)(ix + motion_depth.x * width);
        int32_t py = (int32_t)(iy + motion_depth.y * height);

        bool is_acceptable = AT_MATH_IS_IN_BOUND(px, 0, width - 1)
            && AT_MATH_IS_IN_BOUND(py, 0, height - 1);

        if (is_acceptable)
        {
            aten::LightSampleResult lightsample;

            auto neighbor_idx = getIdx(px, py, width);
            const auto& neighbor_reservoir = prev_reservoirs[neighbor_idx];

            auto m = std::min(neighbor_reservoir.M, maxM);

            if (neighbor_reservoir.IsValid()) {
                const auto& neighbor_info = infos[neighbor_idx];

                const auto& neighbor_normal = neighbor_info.nml;

                aten::MaterialParameter neightbor_mtrl;
                auto is_valid_mtrl = AT_NAME::FillMaterial(
                    neightbor_mtrl,
                    ctxt,
                    neighbor_info.mtrl_idx,
                    neighbor_info.is_voxel);

                const auto neighbor_mesh_id = static_cast<int32_t>(aov_albedo_meshid[neighbor_idx].w);

                // Check how close with neighbor pixel.
                is_acceptable = is_valid_mtrl
                    && _detail::IsAcceptableNeighbor(
                        mtrl, mesh_id, normal,
                        neightbor_mtrl, neighbor_mesh_id, neighbor_normal);

                if (is_acceptable) {
                    const auto light_pos = neighbor_reservoir.y;

                    const auto& light = ctxt.lights[light_pos];

                    AT_NAME::Light::sample(lightsample, light, ctxt, self_info.p, neighbor_normal, &sampler, 0);

                    // Compute target pdf at the center pixel with the output sample in neighor's reservoir.
                    // In this case, "the output sample in neighor's reservoir" mean the sampled light of the neighor pixel.
                    const auto target_pdf = _detail::ComputeTargetPDF(
                        lightsample,
                        light.attrib,
                        self_info.nml, self_info.wi,
                        mtrl,
                        self_info.u, self_info.v,
                        self_info.pre_sampled_r);

                    auto weight = target_pdf * neighbor_reservoir.W * m;

                    auto r = sampler.nextSample();

                    if (combined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                        candidate_target_pdf = target_pdf;
                    }
                }
            }
            else {
                combined_reservoir.update(lightsample, -1, 0.0f, m, 0.0f);
            }
        }

        if (candidate_target_pdf > 0.0f) {
            combined_reservoir.target_pdf_of_y = candidate_target_pdf;
            // NOTE
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
            combined_reservoir.W = combined_reservoir.w_sum / (combined_reservoir.target_pdf_of_y * combined_reservoir.M);
        }

        if (!isfinite(combined_reservoir.W)) {
            combined_reservoir.clear();
        }
    }

    template<class CONTEXT>
    inline AT_DEVICE_API void ApplySpatialReuse(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        CONTEXT& ctxt,
        aten::sampler& sampler,
        const aten::const_span<AT_NAME::Reservoir>& reservoirs,
        aten::span<AT_NAME::Reservoir>& dst_reservoirs,
        const aten::const_span<AT_NAME::ReSTIRInfo>& infos,
        const aten::const_span<AT_NAME::_detail::v4>& aov_albedo_meshid)
    {
        const auto idx = getIdx(ix, iy, width);

        const auto& self_info = infos[idx];

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            self_info.mtrl_idx,
            self_info.is_voxel);

        const auto& normal = self_info.nml;

        const auto mesh_id = static_cast<int32_t>(aov_albedo_meshid[idx].w);

        constexpr int32_t offset_x[] = {
            -1,  0,  1,
            -1,  1,
            -1,  0,  1,
        };
        constexpr int32_t offset_y[] = {
            -1, -1, -1,
             0,  0,
             1,  1,  1,
        };

        auto& combined_reservoir = dst_reservoirs[idx];
        combined_reservoir.clear();

        const auto& reservoir = reservoirs[idx];

        float candidate_target_pdf = 0.0f;

        if (reservoir.IsValid()) {
            combined_reservoir = reservoir;
            candidate_target_pdf = reservoir.target_pdf_of_y;
        }

        int32_t M_sum = combined_reservoir.M;

#pragma unroll
        for (int32_t i = 0; i < AT_COUNTOF(offset_x); i++) {
            const int32_t xx = ix + offset_x[i];
            const int32_t yy = iy + offset_y[i];

            bool is_acceptable = AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
                && AT_MATH_IS_IN_BOUND(yy, 0, height - 1);

            if (is_acceptable)
            {
                aten::LightSampleResult lightsample;

                auto neighbor_idx = getIdx(xx, yy, width);
                const auto& neighbor_reservoir = reservoirs[neighbor_idx];

                if (neighbor_reservoir.IsValid()) {
                    const auto& neighbor_info = infos[neighbor_idx];

                    const auto& neighbor_normal = neighbor_info.nml;

                    aten::MaterialParameter neightbor_mtrl;
                    auto is_valid_mtrl = AT_NAME::FillMaterial(
                        neightbor_mtrl,
                        ctxt,
                        neighbor_info.mtrl_idx,
                        neighbor_info.is_voxel);

                    const auto neighbor_mesh_id = static_cast<int32_t>(aov_albedo_meshid[neighbor_idx].w);

                    // Check how close with neighbor pixel.
                    is_acceptable = is_valid_mtrl
                        && _detail::IsAcceptableNeighbor(
                            mtrl, mesh_id, normal,
                            neightbor_mtrl, neighbor_mesh_id, neighbor_normal);

                    if (is_acceptable) {
                        const auto light_pos = neighbor_reservoir.y;

                        const auto& light = ctxt.lights[light_pos];

                        AT_NAME::Light::sample(lightsample, light, ctxt, self_info.p, neighbor_normal, &sampler, 0);

                        // Compute target pdf at the center pixel with the output sample in neighor's reservoir.
                        // In this case, "the output sample in neighor's reservoir" mean the sampled light of the neighor pixel.
                        const auto target_pdf = _detail::ComputeTargetPDF(
                            lightsample,
                            light.attrib,
                            self_info.nml, self_info.wi,
                            mtrl,
                            self_info.u, self_info.v,
                            self_info.pre_sampled_r);

                        auto m = neighbor_reservoir.M;
                        auto weight = target_pdf * neighbor_reservoir.W * m;

                        auto r = sampler.nextSample();

                        if (combined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                            candidate_target_pdf = target_pdf;
                        }
                    }
                }
                else {
                    combined_reservoir.update(lightsample, -1, 0.0f, neighbor_reservoir.M, 0.0f);
                }
            }
        }

        if (candidate_target_pdf > 0.0f) {
            combined_reservoir.target_pdf_of_y = candidate_target_pdf;
            // NOTE
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
            combined_reservoir.W = combined_reservoir.w_sum / (combined_reservoir.target_pdf_of_y * combined_reservoir.M);
        }

        if (!isfinite(combined_reservoir.W)) {
            combined_reservoir.clear();
        }
    }

    inline AT_DEVICE_API std::optional<aten::vec3> ComputeContribution(
        const Reservoir& reservoir,
        const ReSTIRInfo& restir_info,
        const aten::MaterialParameter& mtrl,
        const AT_NAME::_detail::v4& albedo_meshid,
        const AT_NAME::ShadowRay& shadow_ray,
        const aten::const_span<aten::LightParameter>& lights)
    {
        if (reservoir.IsValid()) {
            const auto& orienting_normal = restir_info.nml;

            const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

            const auto& light = lights[reservoir.y];

            const auto& nmlLight = reservoir.light_sample_.nml;
            const auto& dirToLight = shadow_ray.raydir;
            const auto& distToLight = shadow_ray.distToLight;

            const auto cosShadow = aten::abs(dot(orienting_normal, dirToLight));
            const auto cosLight = aten::abs(dot(nmlLight, -dirToLight));
            const auto dist2 = distToLight * distToLight;

            // TODO
            // 計算済みのalbedoを与えているため
            // u,v は samplePDF/sampleBSDF 内部では利用されていない
            constexpr auto u = 0.0f;
            constexpr auto v = 0.0f;

            const auto le = _detail::ComputeRadiance(
                reservoir.light_sample_, light.attrib,
                orienting_normal, restir_info.wi,
                mtrl,
                u, v, albedo,
                restir_info.pre_sampled_r);

            auto contrib = le * reservoir.W;

            return contrib;
        }

        return std::nullopt;
    }
}
}
