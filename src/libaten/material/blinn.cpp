#include "math/math.h"
#include "material/blinn.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    AT_DEVICE_MTRL_API real MicrofacetBlinn::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        // NOTE
        // http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
        // Half-angle based

        // NOTE
        // https://segmentfault.com/a/1190000000432254

        // half vector.
        auto wh = normalize(-wi + wo);

        auto costheta = dot(normal, wh);

        auto n = param->shininess;

        auto c = dot(wo, wh);

        real pdf = c > AT_MATH_EPSILON ?
            ((n + 1) / (2 * AT_MATH_PI)) * (aten::pow(costheta, n) / (4 * c))
            : 0;

        return pdf;
    }

    AT_DEVICE_MTRL_API real MicrofacetBlinn::pdf(
        const aten::vec3& normal, 
        const aten::vec3& wi,    /* in */
        const aten::vec3& wo,    /* out */
        real u, real v) const
    {
        auto ret = pdf(&m_param, normal, wi, wo, u, v);
        return ret;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        // NOTE
        // http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
        // Lobe Distribution Sampling

        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
        // Sampling Blinn

        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

#if 1
        // Sample halfway vector first, then reflect wi around that
        auto costheta = aten::pow(r1, 1 / (param->shininess + 1));
        auto sintheta = aten::sqrt(1 - costheta * costheta);

        // phi = 2*PI*ksi2
        auto cosphi = aten::cos(AT_MATH_PI_2 * r2);
        auto sinphi = aten::sqrt(real(1) - cosphi * cosphi);
#else
        auto theta = aten::acos(aten::pow(r1, 1 / (m_param.shininess + 1)));
        auto phi = AT_MATH_PI_2 * r2;

        auto costheta = aten::cos(theta);
        auto sintheta = aten::sqrt(1 - costheta * costheta);

        auto cosphi = aten::cos(phi);
        auto sinphi = aten::sqrt(1 - cosphi * cosphi);
#endif

        // Ortho normal base.
        auto n = normal;
        auto t = aten::getOrthoVector(normal);
        auto b = normalize(cross(n, t));

        auto w = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
        w = normalize(w);

        auto dir = wi - 2 * dot(wi, w) * w;

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        auto dir = sampleDirection(&m_param, normal, ray.dir, u, v, sampler);
        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        real fresnel = 1;
        real ior = param->ior;
        real shininess = param->shininess;

        auto albedo = param->baseColor;
        albedo *= sampleTexture(
            param->albedoMap,
            u, v,
            aten::vec3(real(1)));

        auto ret = bsdf(albedo, shininess, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        real fresnel = 1;
        real ior = param->ior;
        real shininess = param->shininess;

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        auto ret = bsdf(albedo, shininess, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        auto ret = bsdf(&m_param, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBlinn::bsdf(
        const aten::vec3& albedo,
        const real shininess,
        const real ior,
        real& fresnel,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        // ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
        real ni = real(1);    // ^‹ó

        // •¨‘Ì“à•”‚Ì‹üÜ—¦.
        real nt = ior;

        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        // TODO
        // Desney‚¾‚Æabs‚µ‚Ä‚È‚¢‚ªAAMD‚Ì‚Í‚µ‚Ä‚¢‚é....
        auto NdotH = aten::abs(dot(N, H));
        auto VdotH = aten::abs(dot(V, H));
        auto NdotL = aten::abs(dot(N, L));
        auto NdotV = aten::abs(dot(N, V));

        auto a = shininess;

        real F(1);
        {
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) à R0 + (1 - R0)(1 - cosƒ¦)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (ni - nt) / (ni + nt);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(L, H));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

        auto denom = 4 * NdotL * NdotV;

        // Compute D.
#if 0
        // NOTE
        // https://www.siggraph.org/education/materials/HyperGraph/illumin/specular_highlights/blinn_model_for_specular_reflect_1.htm
        auto x = aten::acos(NdotH) * a;
        real D = aten::exp(-x * x);
#else
        // NOTE
        // http://simonstechblog.blogspot.jp/2011/12/microfacet-bsdf.html
        real D = (a + 2) / (2 * AT_MATH_PI);
        D *= aten::pow(aten::cmpMax((real)0, NdotH), a);
#endif

        // Compute G.
        real G(1);
        {
            // Cook-Torrance geometry function.
            // http://simonstechblog.blogspot.jp/2011/12/microfacet-bsdf.html

            auto G1 = 2 * NdotH * NdotL / VdotH;
            auto G2 = 2 * NdotH * NdotV / VdotH;
            G = aten::cmpMin((real)1, aten::cmpMin(G1, G2));
        }

        auto bsdf = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : aten::vec3(0);

        fresnel = F;

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API void MicrofacetBlinn::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler);

#if 1
        result->pdf = pdf(param, normal, wi, result->dir, u, v);

        real fresnel = 1;
        real ior = param->ior;
        real shininess = param->shininess;

        auto albedo = param->baseColor;
        albedo *= sampleTexture(
            param->albedoMap,
            u, v,
            aten::vec3(real(1)));

        result->bsdf = bsdf(albedo, shininess, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
#else
        aten::vec3 V = -in;
        aten::vec3 L = ret.dir;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        auto NdotL = aten::abs(dot(N, L));
        auto NdotH = aten::abs(dot(N, H));

        if (NdotL > 0 && NdotH > 0) {
            ret.pdf = pdf(normal, in, ret.dir, u, v);

            ret.bsdf = bsdf(normal, in, ret.dir, u, v);
        }
        else {
            ret.pdf = 0;
        }
#endif
    }

    AT_DEVICE_MTRL_API void MicrofacetBlinn::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        const aten::vec3& externalAlbedo,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler);

        result->pdf = pdf(param, normal, wi, result->dir, u, v);

        real fresnel = 1;
        real ior = param->ior;
        real shininess = param->shininess;

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        result->bsdf = bsdf(albedo, shininess, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API MaterialSampling MicrofacetBlinn::sample(
        const aten::ray& ray,
        const aten::vec3& normal,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/) const
    {
        MaterialSampling ret;

        sample(
            &ret,
            &m_param,
            normal,
            ray.dir,
            orgnormal,
            sampler,
            u, v,
            isLightPath);
        
        return std::move(ret);
    }

    bool MicrofacetBlinn::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, shininess, real(0), real(1000));
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1 || b2;
    }
}
