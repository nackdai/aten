#include "scene/scene.h"
#include "misc/color.h"
#include "geometry/transformable.h"
#include "light/light_impl.h"

namespace aten {
    std::shared_ptr<Light> scene::sampleLight(
        const context& ctxt,
        const vec3& org,
        const vec3& nml,
        sampler* sampler,
        real& selectPdf,
        LightSampleResult& sampleRes)
    {
        auto num = ctxt.GetLightNum();

        if (num > 0) {
            auto r = sampler->nextSample();
            uint32_t idx = (uint32_t)aten::clamp<real>(r * num, 0, num - 1);
            const auto light = ctxt.GetLightInstance(idx);

            const auto& light_param = light->param();

            Light::sample(sampleRes, light_param, ctxt, org, nml, sampler);
            selectPdf = real(1) / num;

            return light;
        }

        selectPdf = 1;
        return nullptr;
    }

    struct Reservoir {
        float w_sum_{ 0.0f };
        float sample_weight_{ 0.0f };
        uint32_t m_{ 0 };
        int32_t light_idx_{ 0 };
        float pdf_{ 0.0f };
        float target_density_{ 0.0f };
        aten::LightSampleResult light_sample_;

        void clear()
        {
            w_sum_ = 0.0f;
            sample_weight_ = 0.0f;
            m_ = 0;
            light_idx_ = -1;
            pdf_ = 0.0f;
            target_density_ = 0.0f;
        }

        bool update(
            const aten::LightSampleResult& light_sample,
            int32_t new_target_idx, float weight, float u)
        {
            w_sum_ += weight;
            bool is_accepted = u < weight / w_sum_;
            if (is_accepted) {
                light_sample_ = light_sample;
                light_idx_ = new_target_idx;
                sample_weight_ = weight;
            }
            m_++;
            return is_accepted;
        }
    };

    std::shared_ptr<Light> scene::sampleLightWithReservoir(
        const aten::context& ctxt,
        const aten::vec3& org,
        const aten::vec3& nml,
        std::function<aten::vec3(const aten::vec3&)> compute_brdf,
        aten::sampler* sampler,
        real& selectPdf,
        aten::LightSampleResult& sampleRes)
    {
        // Resampled Importance Sampling.

        static constexpr auto MaxLightCount = 32U;

        const auto max_light_num = static_cast<decltype(MaxLightCount)>(ctxt.GetLightNum());
        const auto light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

        Reservoir reservoir;
        reservoir.clear();

        real selected_target_density = real(0);

        real lightSelectProb = real(1) / max_light_num;

        for (auto i = 0U; i < light_cnt; i++) {
            const auto r_light = sampler->nextSample();
            const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

            const auto light = ctxt.GetLightInstance(light_pos);
            const auto& light_param = light->param();

            aten::LightSampleResult lightsample;
            Light::sample(lightsample, light_param, ctxt, org, nml, sampler);

            aten::vec3 nmlLight = lightsample.nml;
            aten::vec3 dirToLight = normalize(lightsample.dir);

            auto brdf = compute_brdf(dirToLight);

            auto cosShadow = dot(nml, dirToLight);
            auto cosLight = dot(nmlLight, -dirToLight);
            auto dist2 = aten::squared_length(lightsample.dir);

            auto energy = brdf * lightsample.finalColor;

            cosShadow = aten::abs(cosShadow);

            if (cosShadow > 0 && cosLight > 0) {
                if (light->isSingular()) {
                    energy = energy * cosShadow;
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
            // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xz) * M)
            reservoir.pdf_ = reservoir.w_sum_ / (reservoir.target_density_ * reservoir.m_);
        }

        if (!std::isfinite(reservoir.pdf_)) {
            reservoir.pdf_ = 0.0f;
            reservoir.target_density_ = 0.0f;
            reservoir.light_idx_ = -1;
        }

        if (reservoir.light_idx_ < 0) {
            return nullptr;
        }

        selectPdf = reservoir.pdf_;
        sampleRes = reservoir.light_sample_;

        return ctxt.GetLightInstance(reservoir.light_idx_);
    }

    void scene::render(
        aten::hitable::FuncPreDraw func,
        std::function<bool(const std::shared_ptr<aten::hitable>&)> funcIfDraw,
        const context& ctxt) const
    {
        uint32_t triOffset = 0;

        for (auto h : m_list) {
            bool willDraw = funcIfDraw ? funcIfDraw(h) : true;

            if (willDraw) {
                h->render(func, ctxt, aten::mat4::Identity, aten::mat4::Identity, -1, triOffset);
            }

            auto item = h->getHasObject();
            if (item) {
                triOffset += item->getTriangleCount();
            }
        }
    }
}
