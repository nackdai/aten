#include "material/toon.h"

namespace AT_NAME
{
    aten::vec3 toon::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        // TODO
#if 0
        auto* light = getTargetLight();
        AT_ASSERT(light);

        auto res = light->sample(ray.org, sampler);

        return res.dir;
#else
        AT_ASSERT(false);
        return aten::vec3(0);
#endif
    }

    aten::vec3 toon::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        real cosShadow = dot(normal, wo);
        auto ret = bsdf(cosShadow, u, v);
        return ret;
    }

    MaterialSampling toon::sample(
        const aten::ray& ray,
        const aten::vec3& normal,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
        real u, real v,
        bool isLightPath/*= false*/) const
    {
        MaterialSampling ret;

        const aten::vec3& in = ray.dir;

        ret.dir = sampleDirection(ray, normal, u, v, sampler);
        ret.pdf = pdf(normal, in, ret.dir, u, v);
        ret.bsdf = bsdf(normal, in, ret.dir, u, v);

        ret.fresnel = 1;

        return ret;
    }

    aten::vec3 toon::bsdf(
        real cosShadow,
        real u, real v) const
    {
        const real c = aten::clamp<real>(cosShadow, 0, 1);

        aten::vec4 albedo = color();
        albedo *= sampleAlbedoMap(u, v);

        real coeff = 1;

        if (m_func) {
            coeff = m_func(c);
        }
        else {
            // 適当...
            if (c < 0.5) {
                coeff = 0;
            }
            else {
                coeff = 1;
            }
        }

        // TODO
        // 完全に影にならないようにしてみる.
        coeff = std::max<real>(coeff, real(0.05));
        //coeff = aten::clamp<real>(coeff, 0.05, 1);

        aten::vec3 bsdf = albedo * coeff;

        return bsdf;
    }
}
