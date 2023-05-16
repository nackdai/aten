#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_params.h"
#include "kernel/material.cuh"
#include "kernel/light.cuh"

#include "reservior.h"

inline __device__ int32_t sampleLightWithReservoirRIP(
    idaten::Reservoir& reservoir,
    const aten::MaterialParameter& mtrl,
    idaten::context* ctxt,
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

    const auto max_light_num = static_cast<decltype(MaxLightCount)>(ctxt->lightnum);
    const auto light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

    reservoir.clear();

    real selected_target_density = real(0);

    real lightSelectProb = real(1) / max_light_num;

    for (auto i = 0U; i < light_cnt; i++) {
        const auto r_light = sampler->nextSample();
        const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

        const auto& light = ctxt->lights[light_pos];

        aten::LightSampleResult lightsample;
        sampleLight(&lightsample, ctxt, &light, org, normal, sampler, lod);

        aten::vec3 nmlLight = lightsample.nml;
        aten::vec3 dirToLight = normalize(lightsample.dir);

        auto pdf = AT_NAME::material::samplePDF(&mtrl, normal, ray_dir, dirToLight, u, v);
        auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(&mtrl, normal, ray_dir, dirToLight, u, v, albedo, pre_sampled_r);
        brdf /= pdf;

        auto cosShadow = dot(normal, dirToLight);
        auto cosLight = dot(nmlLight, -dirToLight);
        auto dist2 = aten::squared_length(lightsample.dir);

        auto energy = brdf * lightsample.finalColor;

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
