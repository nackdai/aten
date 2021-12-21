#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_params.h"
#include "kernel/material.cuh"
#include "kernel/light.cuh"

#include "reservior.h"

class ComputeBrdfFunctor {
public:
    __device__ ComputeBrdfFunctor(
        idaten::Context& ctxt,
        const aten::MaterialParameter& mtrl,
        const aten::vec3& orienting_normal,
        const aten::vec3& ray_dir,
        float u, float v,
        const aten::vec4& albedo)
        : ctxt_(ctxt), mtrl_(mtrl), orienting_normal_(orienting_normal),
        ray_dir_(ray_dir), u_(u), v_(v), albedo_(albedo) {}

    inline __device__ aten::vec3 operator()(const aten::vec3& dir_to_light) {
        return sampleBSDF(
            &ctxt_, &mtrl_, orienting_normal_, ray_dir_, dir_to_light, u_, v_, albedo_);
    }

private:
    idaten::Context& ctxt_;
    const aten::MaterialParameter& mtrl_;
    const aten::vec3& orienting_normal_;
    const aten::vec3& ray_dir_;
    float u_;
    float v_;
    const aten::vec4& albedo_;
};

inline __device__ void computeLighting(
    aten::vec3& energy,
    const aten::LightParameter& light,
    const aten::vec3& normal,
    const aten::vec3& nmlLight,
    float pdfLight,
    const aten::vec3& light_color,
    const aten::vec3& dirToLight,
    float distToLight)
{
    auto cosShadow = dot(normal, dirToLight);

    energy.r = 0.0f;
    energy.g = 0.0f;
    energy.b = 0.0f;

    if (cosShadow > 0) {
        if (light.attrib.isInfinite) {
            energy = light_color * cosShadow;
        }
        else {
            auto cosLight = dot(nmlLight, -dirToLight);

            if (cosLight > 0) {
                if (light.attrib.isSingular) {
                    energy = light_color * cosShadow * cosLight;
                }
                else {
                    auto dist2 = distToLight * distToLight;
                    energy = light_color * cosShadow * cosLight / dist2;
                }
            }
        }

        energy /= pdfLight;
    }
}

inline __device__ int sampleLightWithReservoirRIP(
    idaten::Reservoir& reservoir,
    aten::LightParameter* target_light,
    ComputeBrdfFunctor& compute_brdf,
    idaten::Context* ctxt,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int lod = 0)
{
    constexpr auto MaxLightCount = 32U;

    const auto max_light_num = static_cast<decltype(MaxLightCount)>(ctxt->lightnum);
    const auto max_light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

    reservoir.clear();

    real selected_target_density = real(0);

    for (auto i = 0U; i < max_light_cnt; i++) {
        const auto r_light = sampler->nextSample();
        const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

        const auto& light = ctxt->lights[light_pos];

        aten::LightSampleResult lightsample;
        sampleLight(&lightsample, ctxt, &light, org, normal, sampler, lod);

        aten::vec3 nmlLight = lightsample.nml;
        aten::vec3 dirToLight = normalize(lightsample.dir);

        auto brdf = compute_brdf(dirToLight);

        auto cosShadow = dot(normal, dirToLight);
        auto cosLight = dot(nmlLight, -dirToLight);
        auto dist2 = aten::squared_length(lightsample.dir);

        auto energy = brdf * lightsample.finalColor;

        cosLight = aten::abs(cosLight);

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
        auto sampling_density = lightsample.pdf;    // q

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
