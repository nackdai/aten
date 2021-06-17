#include "scene/scene.h"
#include "misc/color.h"
#include "geometry/transformable.h"

namespace aten {
    bool scene::hitLight(
        const context& ctxt,
        const Light* light,
        const vec3& lightPos,
        const ray& r,
        real t_min, real t_max,
        hitrecord& rec)
    {
        Intersection isect;
        bool isHit = this->hit(ctxt, r, t_min, t_max, rec, isect);

        real distToLight = length(lightPos - r.org);
        real distHitObjToRayOrg = length(rec.p - r.org);

        const auto& param = light->param();

        const auto lightobj = param.objid >= 0 ? ctxt.getTransformable(param.objid) : nullptr;
        const auto hitobj = isect.objid >= 0 ? ctxt.getTransformable(isect.objid) : nullptr;

        isHit = scene::hitLight(
            isHit,
            param.attrib,
            lightobj.get(),
            distToLight,
            distHitObjToRayOrg,
            isect.t,
            hitobj.get());

        return isHit;
    }

    std::shared_ptr<Light> scene::sampleLight(
        const context& ctxt,
        const vec3& org,
        const vec3& nml,
        sampler* sampler,
        real& selectPdf,
        LightSampleResult& sampleRes)
    {
        auto num = m_lights.size();

        if (num > 0) {
            auto r = sampler->nextSample();
            uint32_t idx = (uint32_t)aten::clamp<real>(r * num, 0, num - 1);
            const auto& light = m_lights[idx];

            sampleRes = light->sample(ctxt, org, nml, sampler);
            selectPdf = real(1) / num;

            return light;
        }

        selectPdf = 1;
        return nullptr;
    }

    std::shared_ptr<Light> scene::sampleLight(
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

        const auto max_light_num = static_cast<decltype(MaxLightCount)>(m_lights.size());
        const auto light_cnt = aten::cmpMin(MaxLightCount, max_light_num);

#if 1
        // Reservoir

        auto r = sampler->nextSample();
        auto w_sum = real(0);

        int32_t selected_light_idx = -1;
        real selected_cost = real(0);

        for (auto i = 0U; i < light_cnt; i++) {
            const auto r_light = sampler->nextSample();
            const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

            const auto& light = m_lights[light_pos];

            const auto lightsample = light->sample(ctxt, org, nml, sampler);

            vec3 posLight = lightsample.pos;
            vec3 nmlLight = lightsample.nml;
            real pdfLight = lightsample.pdf;
            vec3 dirToLight = normalize(lightsample.dir);

            auto brdf = compute_brdf(dirToLight);

            auto cosShadow = dot(nml, dirToLight);
            auto dist2 = squared_length(lightsample.dir);

            auto light_energy = color::luminance(lightsample.le);
            auto brdf_energy = color::luminance(brdf);

            auto energy = brdf_energy * light_energy;

            real cost = real(0);

            if (cosShadow > 0) {
                if (light->isInfinite()) {
                    cost = energy * cosShadow / pdfLight;
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);
                    cost = energy * cosShadow * cosLight / dist2 / pdfLight;
                }
            }

            if (cost > 0) {
                w_sum += cost;

                if (r < cost / w_sum) {
                    selected_light_idx = light_pos;
                    sampleRes = lightsample;
                    selected_cost = cost;
                }
            }
        }

        if (selected_cost > 0) {
            // NOTE
            // ò_ï∂ìIÇ…ÇÕÅAw_sum / M Çä|ÇØÇÈÇÃÇ…ëŒÇµÇƒ
            // path tracingÇÃì‡ïîÇ≈ÇÕÅA1 / lightSelectPdf Ç≈åvéZÇµÇƒÇ¢ÇÈÇÃÇ≈
            // Ç±Ç±Ç≈ÇÕãtêîÇ…ÇµÇƒÇ®Ç≠Ç±Ç∆Ç≈ÅAç≈èIìIÇ…ä|ÇØéZÇ…Ç»ÇÈÇÊÇ§Ç…Ç∑ÇÈ.
            selectPdf = light_cnt / w_sum;

            return m_lights[selected_light_idx];;
        }

        return nullptr;
#else
        std::vector<LightSampleResult> samples(m_lights.size());
        std::vector<real> costs(m_lights.size());

        real sumCost = 0;

        for (auto i = 0U; i < light_cnt; i++) {
            const auto r_light = sampler->nextSample();
            const auto light_pos = aten::clamp<decltype(max_light_num)>(r_light * max_light_num, 0, max_light_num - 1);

            const auto light = m_lights[light_pos];

            samples[i] = light->sample(ctxt, org, nml, sampler);

            const auto& lightsample = samples[i];

            vec3 posLight = lightsample.pos;
            vec3 nmlLight = lightsample.nml;
            real pdfLight = lightsample.pdf;
            vec3 dirToLight = normalize(lightsample.dir);

            auto brdf = compute_brdf(dirToLight);

            auto cosShadow = dot(nml, dirToLight);
            auto dist2 = squared_length(lightsample.dir);

            auto light_energy = color::luminance(lightsample.finalColor);
            auto brdf_energy = color::luminance(brdf);

            auto energy = brdf_energy * light_energy;

            real cost = real(0);

            if (cosShadow > 0) {
                if (light->isInfinite()) {
                    cost = energy * cosShadow / pdfLight;
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);

                    if (light->isSingular()) {
                        cost = energy * cosShadow * cosLight / pdfLight;
                    }
                    else {
                        cost = energy * cosShadow * cosLight / dist2 / pdfLight;
                    }
                }
            }

            costs[i] = cost;
            sumCost += cost;
        }

        auto r = sampler->nextSample() * sumCost;

        real sum = 0;

        for (int i = 0; i < costs.size(); i++) {
            const auto c = costs[i];
            sum += c;

            if (r <= sum && c > 0) {
                auto light = m_lights[i];
                sampleRes = samples[i];
                selectPdf = c / sumCost;
                return light;
            }
        }

        return nullptr;
#endif
    }

    void scene::drawForGBuffer(
        aten::hitable::FuncPreDraw func,
        std::function<bool(const std::shared_ptr<aten::hitable>&)> funcIfDraw,
        const context& ctxt) const
    {
        uint32_t triOffset = 0;

        for (auto h : m_list) {
            bool willDraw = funcIfDraw ? funcIfDraw(h) : true;

            if (willDraw) {
                h->drawForGBuffer(func, ctxt, aten::mat4::Identity, aten::mat4::Identity, -1, triOffset);
            }

            auto item = h->getHasObject();
            if (item) {
                triOffset += item->getTriangleCount();
            }
        }
    }
}
