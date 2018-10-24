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

        auto lightobj = param.objid >= 0 ? ctxt.getTransformable(param.objid) : nullptr;
        auto hitobj = isect.objid >= 0 ? ctxt.getTransformable(isect.objid) : nullptr;

        isHit = scene::hitLight(
            isHit,
            param.attrib,
            lightobj,
            distToLight,
            distHitObjToRayOrg,
            isect.t,
            hitobj);

        return isHit;
    }

    Light* scene::sampleLight(
        const context& ctxt,
        const vec3& org,
        const vec3& nml,
        sampler* sampler,
        real& selectPdf,
        LightSampleResult& sampleRes)
    {
#if 1
        Light* light = nullptr;

        auto num = m_lights.size();
        if (num > 0) {
            auto r = sampler->nextSample();
            uint32_t idx = (uint32_t)aten::clamp<real>(r * num, 0, num - 1);
            light = m_lights[idx];

            sampleRes = light->sample(ctxt, org, nml, sampler);
            selectPdf = real(1) / num;
        }
        else {
            selectPdf = 1;
        }

        return light;
#else
        // Resampled Importance Sampling.
        // For reducing variance...maybe...

        std::vector<LightSampleResult> samples(m_lights.size());
        std::vector<real> costs(m_lights.size());

        real sumCost = 0;

        for (int i = 0; i < m_lights.size(); i++) {
            const auto light = m_lights[i];

            samples[i] = light->sample(org, nml, sampler);

            const auto& lightsample = samples[i];

            vec3 posLight = lightsample.pos;
            vec3 nmlLight = lightsample.nml;
            real pdfLight = lightsample.pdf;
            vec3 dirToLight = normalize(lightsample.dir);

            auto cosShadow = dot(nml, dirToLight);
            auto dist2 = squared_length(lightsample.dir);
            auto dist = aten::sqrt(dist2);

            auto illum = color::luminance(lightsample.finalColor);

            if (cosShadow > 0) {
                if (light->isSingular() || light->isInfinite()) {
                    costs[i] = illum * cosShadow / pdfLight;
                }
                else {
                    costs[i] = illum * cosShadow / dist2 / pdfLight;
                }
                sumCost += costs[i];
            }
            else {
                costs[i] = 0;
            }
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
        std::function<bool(aten::hitable*)> funcIfDraw,
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
