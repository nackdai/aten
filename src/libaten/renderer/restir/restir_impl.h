#pragma once

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
    template<class CONTEXT>
    inline AT_DEVICE_API int32_t SampleLightWithReservoirRIP(
        AT_NAME::Reservoir& reservoir,
        const aten::MaterialParameter& mtrl,
        const CONTEXT& ctxt,
        const aten::vec3& org,
        const aten::vec3& normal,
        const aten::vec3& ray_dir,
        float u, float v,
        const aten::vec4& albedo,
        aten::sampler* sampler,
        real pre_sampled_r,
        int32_t lod = 0)
    {
        constexpr auto MaxLightCount = 32U;

        const auto light_num = ctxt.GetLightNum();
        const auto max_light_num = static_cast<decltype(MaxLightCount)>(light_num);
        const auto light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

        reservoir.clear();

        real selected_target_density = real(0);

        real lightSelectProb = real(1) / max_light_num;

        for (auto i = 0U; i < light_cnt; i++) {
            const auto r_light = sampler->nextSample();
            const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

            const auto& light = ctxt.GetLight(light_pos);

            aten::LightSampleResult lightsample;
            AT_NAME::Light::sample(lightsample, light, ctxt, org, normal, sampler, lod);

            aten::vec3 nmlLight = lightsample.nml;
            aten::vec3 dirToLight = normalize(lightsample.dir);

            auto pdf = AT_NAME::material::samplePDF(&mtrl, normal, ray_dir, dirToLight, u, v);
            auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(&mtrl, normal, ray_dir, dirToLight, u, v, albedo, pre_sampled_r);
            brdf /= pdf;

            auto cosShadow = dot(normal, dirToLight);
            auto cosLight = dot(nmlLight, -dirToLight);
            auto dist2 = aten::squared_length(lightsample.dir);

            auto energy = brdf * lightsample.light_color;

            cosShadow = aten::abs(cosShadow);

            if (cosShadow > 0 && cosLight > 0) {
                if (light.attrib.isSingular) {
                    energy = energy * cosShadow * cosLight;
                }
                else {
                    energy = energy * cosShadow * cosLight / dist2;
                }
            }
            else {
                energy.x = energy.y = energy.z = 0.0f;
            }

            auto target_density = (energy.x + energy.y + energy.z) / 3; // p_hat
            auto sampling_density = lightsample.pdf * lightSelectProb;  // q

            // NOTE
            // p_hat(xi) / q(xi)
            auto weight = sampling_density > 0
                ? target_density / sampling_density
                : 0.0f;

            auto r = sampler->nextSample();

            if (reservoir.update(lightsample, light_pos, weight, r)) {
                selected_target_density = target_density;
            }
        }

        if (selected_target_density > 0.0f) {
            reservoir.target_density_ = selected_target_density;
            // NOTE
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
            reservoir.pdf_ = reservoir.w_sum_ / (reservoir.target_density_ * reservoir.m_);
        }

        if (!isfinite(reservoir.pdf_)) {
            reservoir.pdf_ = 0.0f;
            reservoir.target_density_ = 0.0f;
            reservoir.light_idx_ = -1;
        }

        return reservoir.light_idx_;
    }

    template <class CONTEXT>
    inline AT_DEVICE_API void EvaluateVisibility(
        int32_t idx,
        int32_t bounce,
        idaten::Path& paths,
        CONTEXT& ctxt,
        aten::span<idaten::Reservoir>& reservoirs,
        const aten::const_span<idaten::ReSTIRInfo>& restir_infos,
        aten::span<idaten::ShadowRay>& shadowRays)
    {
        bool isHit = false;

        const auto& reservoir = reservoirs[idx];

        if (reservoir.IsValid()) {
            const auto& restir_info = restir_infos[idx];

            shadowRays[idx].rayorg = restir_info.p + AT_MATH_EPSILON * restir_info.nml;
            shadowRays[idx].raydir = reservoir.light_sample_.pos - shadowRays[idx].rayorg;
            shadowRays[idx].targetLightId = reservoir.light_idx_;
            shadowRays[idx].isActive = true;

            auto dist = length(shadowRays[idx].raydir);;
            shadowRays[idx].distToLight = dist;
            shadowRays[idx].raydir /= dist;

            isHit = AT_NAME::HitShadowRay(idx, bounce, ctxt, paths, shadowRays[idx]);
        }

        if (!isHit) {
            reservoirs[idx].w_sum_ = 0.0f;
            reservoirs[idx].pdf_ = 0.0f;
            reservoirs[idx].target_density_ = 0.0f;
            reservoirs[idx].light_idx_ = -1;
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
        const AT_NAME::_detail::v4& albedo_meshid,
        BufferForMotionDepth& motion_detph_buffer)
    {
        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            self_info.mtrl_idx,
            self_info.is_voxel);

        const auto& normal = self_info.nml;

        const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

        float selected_target_density = combined_reservoir.IsValid()
            ? combined_reservoir.target_density_
            : 0.0f;

        // NOTE
        // In this case, self reservoir's M should be number of number of light sampling.
        const auto maxM = 20 * combined_reservoir.m_;

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

            auto m = std::min(neighbor_reservoir.m_, maxM);

            if (neighbor_reservoir.IsValid()) {
                const auto& neighbor_info = infos[neighbor_idx];

                const auto& neighbor_normal = neighbor_info.nml;

                aten::MaterialParameter neightbor_mtrl;
                auto is_valid_mtrl = AT_NAME::FillMaterial(
                    neightbor_mtrl,
                    ctxt,
                    neighbor_info.mtrl_idx,
                    neighbor_info.is_voxel);

                // Check how close with neighbor pixel.
                is_acceptable = is_valid_mtrl
                    && (mtrl.type == neightbor_mtrl.type)
                    && (dot(normal, neighbor_normal) >= 0.95f);

                if (is_acceptable) {
                    const auto light_pos = neighbor_reservoir.light_idx_;

                    const auto& light = ctxt.lights[light_pos];

                    AT_NAME::Light::sample(lightsample, light, ctxt, self_info.p, neighbor_normal, &sampler, 0);

                    aten::vec3 nmlLight = lightsample.nml;
                    aten::vec3 dirToLight = normalize(lightsample.dir);

                    auto pdf = AT_NAME::material::samplePDF(
                        &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v);
                    auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                        &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v,
                        albedo,
                        self_info.pre_sampled_r);
                    brdf /= pdf;

                    auto cosShadow = dot(normal, dirToLight);
                    auto cosLight = dot(nmlLight, -dirToLight);
                    auto dist2 = aten::squared_length(lightsample.dir);

                    auto energy = brdf * lightsample.light_color;

                    cosShadow = aten::abs(cosShadow);

                    if (cosShadow > 0 && cosLight > 0) {
                        if (light.attrib.isSingular) {
                            energy = energy * cosShadow * cosLight;
                        }
                        else {
                            energy = energy * cosShadow * cosLight / dist2;
                        }
                    }
                    else {
                        energy.x = energy.y = energy.z = 0.0f;
                    }

                    auto target_density = (energy.x + energy.y + energy.z) / 3; // p_hat

                    auto weight = target_density * neighbor_reservoir.pdf_ * m;

                    auto r = sampler.nextSample();

                    if (combined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                        selected_target_density = target_density;
                    }
                }
            }
            else {
                combined_reservoir.update(lightsample, -1, 0.0f, m, 0.0f);
            }
        }

        if (selected_target_density > 0.0f) {
            combined_reservoir.target_density_ = selected_target_density;
            // NOTE
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
            combined_reservoir.pdf_ = combined_reservoir.w_sum_ / (combined_reservoir.target_density_ * combined_reservoir.m_);
        }

        if (!isfinite(combined_reservoir.pdf_)) {
            combined_reservoir.clear();
        }
    }

    template<class CONTEXT>
    inline void AT_DEVICE_API ApplySpatialReuse(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        CONTEXT& ctxt,
        aten::sampler& sampler,
        const aten::const_span<AT_NAME::Reservoir>& reservoirs,
        aten::span<AT_NAME::Reservoir>& dst_reservoirs,
        const aten::const_span<AT_NAME::ReSTIRInfo>& infos,
        const AT_NAME::_detail::v4& albedo_meshid)
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

        const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

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

        auto& comibined_reservoir = dst_reservoirs[idx];
        comibined_reservoir.clear();

        const auto& reservoir = reservoirs[idx];

        float selected_target_density = 0.0f;

        if (reservoir.IsValid()) {
            comibined_reservoir = reservoir;
            selected_target_density = reservoir.target_density_;
        }

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

                    // Check how close with neighbor pixel.
                    is_acceptable = is_valid_mtrl
                        && (mtrl.type == neightbor_mtrl.type)
                        && (dot(normal, neighbor_normal) >= 0.95f);

                    if (is_acceptable) {
                        const auto light_pos = neighbor_reservoir.light_idx_;

                        const auto& light = ctxt.lights[light_pos];

                        AT_NAME::Light::sample(lightsample, light, ctxt, self_info.p, neighbor_normal, &sampler, 0);

                        aten::vec3 nmlLight = lightsample.nml;
                        aten::vec3 dirToLight = normalize(lightsample.dir);

                        auto pdf = AT_NAME::material::samplePDF(
                            &neightbor_mtrl,
                            normal,
                            self_info.wi, dirToLight,
                            self_info.u, self_info.v);
                        auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                            &neightbor_mtrl,
                            normal,
                            self_info.wi, dirToLight,
                            self_info.u, self_info.v,
                            albedo,
                            self_info.pre_sampled_r);
                        brdf /= pdf;

                        auto cosShadow = dot(normal, dirToLight);
                        auto cosLight = dot(nmlLight, -dirToLight);
                        auto dist2 = aten::squared_length(lightsample.dir);

                        auto energy = brdf * lightsample.light_color;

                        cosShadow = aten::abs(cosShadow);

                        if (cosShadow > 0 && cosLight > 0) {
                            if (light.attrib.isSingular) {
                                energy = energy * cosShadow * cosLight;
                            }
                            else {
                                energy = energy * cosShadow * cosLight / dist2;
                            }
                        }
                        else {
                            energy.x = energy.y = energy.z = 0.0f;
                        }

                        auto target_density = (energy.x + energy.y + energy.z) / 3; // p_hat

                        auto m = neighbor_reservoir.m_;
                        auto weight = target_density * neighbor_reservoir.pdf_ * m;

                        auto r = sampler.nextSample();

                        if (comibined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                            selected_target_density = target_density;
                        }
                    }
                }
                else {
                    comibined_reservoir.update(lightsample, -1, 0.0f, neighbor_reservoir.m_, 0.0f);
                }
            }
        }

        if (selected_target_density > 0.0f) {
            comibined_reservoir.target_density_ = selected_target_density;
            // NOTE
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
            comibined_reservoir.pdf_ = comibined_reservoir.w_sum_ / (comibined_reservoir.target_density_ * comibined_reservoir.m_);
        }

        if (!isfinite(comibined_reservoir.pdf_)) {
            comibined_reservoir.clear();
        }
    }

}
}
