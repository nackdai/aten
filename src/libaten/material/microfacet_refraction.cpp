#include "material/microfacet_refraction.h"
#include "material/ggx.h"
#include "material/sample_texture.h"

//#pragma optimize( "", off)

namespace AT_NAME
{
    AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        auto ret = pdf(roughness.r, param->ior, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        aten::vec3 dir = sampleDirection(roughness.r, param->ior, normal, wi, sampler);

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
        real roughness,
        real ior,
        const aten::vec3& nml,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto& in = -wi;
        const auto& out = wo;
        auto n = nml;

        auto NdotI = dot(nml, in);
        auto NdotO = dot(nml, out);

        if (NdotI * NdotO >= 0) {
            // Incomling ray and Outgoing ray are same direction.
            return real(0);
        }

        bool into = (NdotI >= real(0));

        real etai = real(1);    // ê^ãÛÇÃã¸ê‹ó¶.
        real etat = ior;        // ï®ëÃì‡ïîÇÃã¸ê‹ó¶.

        if (!into) {
            auto tmp = etai;
            etai = etat;
            etat = tmp;

            n = -n;
        }

        // Expression(16)
        const auto ht = -(etai * in + etat * out);
        
        const auto wh = normalize(ht);

        auto OdotWh = aten::abs(dot(out, wh));

        // Expression(17)
        auto wh_wo = etai * etai * OdotWh;
        auto denom = dot(ht, ht);

        // Expression(24)
        auto pdf = MicrofacetGGX::sampleGGX_D(wh, n, roughness) * aten::abs(dot(wh, n));

        // Expression(38)
        pdf = denom > AT_MATH_EPSILON ? pdf * wh_wo / denom : real(0);

        return pdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
        real roughness,
        real ior,
        const aten::vec3& wi,
        const aten::vec3& nml,
        aten::sampler* sampler)
    {
        const auto& in = -wi;
        auto n = nml;

        auto NdotI = dot(nml, in);

        if (NdotI == 0) {
            return std::move(aten::vec3(0));
        }

        bool into = (NdotI >= real(0));

        real etai = real(1);    // ê^ãÛÇÃã¸ê‹ó¶.
        real etat = ior;        // ï®ëÃì‡ïîÇÃã¸ê‹ó¶.
        real sign = real(1);

        if (!into) {
            auto tmp = etai;
            etai = etat;
            etat = tmp;

            sign = -sign;

            n = -n;
        }

        // Sample microfacet normal.
        const auto m = MicrofacetGGX::sampleNormal(roughness, n, sampler);

        // Expression(40)
        const auto c = dot(in, m);
        const auto eta = etai / etat;

        //const auto d = 1 + eta * (c * c - 1);
        const auto d = 1 - eta * eta * aten::cmpMax(real(0), 1 - c * c);

        if (d <= 0) {
            return std::move(aten::vec3(0));
        }

        const auto t = (eta * c - aten::sqrt(d));
        auto wo = t * m - eta * in;
        wo = normalize(wo);

        return std::move(wo);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
        const aten::vec3& albedo,
        const real roughness,
        const real ior,
        real& fresnel,
        const aten::vec3& nml,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        const auto& in = -wi;
        const auto& out = wo;
        auto n = nml;

        auto NdotI = dot(nml, in);
        auto NdotO = dot(nml, out);

        if (NdotI * NdotO >= 0) {
            // Incomling ray and Outgoing ray are same direction.
            return std::move(aten::vec3(0));
        }

        bool into = (NdotI >= real(0));

        real etai = real(1);    // ê^ãÛÇÃã¸ê‹ó¶.
        real etat = ior;        // ï®ëÃì‡ïîÇÃã¸ê‹ó¶.

        if (!into) {
            auto tmp = etai;
            etai = etat;
            etat = tmp;

            n = -n;
        }

        const auto ht = -(etai * in + etat * out);
        const auto wh = normalize(ht);

        real D = MicrofacetGGX::sampleGGX_D(wh, n, roughness);

        // Compute G.
        real G(1);
        {
            auto G1_lh = MicrofacetGGX::computeGGXSmithG1(roughness, in, wh);
            auto G1_vh = MicrofacetGGX::computeGGXSmithG1(roughness, out, wh);

            G = G1_lh * G1_vh;
        }

        real F(1);
        {
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) Å‡ R0 + (1 - R0)(1 - cosÉ¶)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (etai - etat) / (etai + etat);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(out, wh));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

        auto IdotN = aten::abs(dot(in, n));
        auto OdotN = aten::abs(dot(out, n));

        auto IdotH = aten::abs(dot(in, wh));
        auto OdotH = aten::abs(dot(out, wh));

        auto denom = dot(ht, ht);    // Expression(17)
        denom *= (IdotN * OdotN);

        // Expression(21)
        auto bsdf = denom > AT_MATH_EPSILON
            ? albedo * (1 - F) * G * D * (IdotH * OdotH) * etat * etat / denom
            : aten::vec3(0);

        fresnel = F;

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API void MicrofacetRefraction::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        result->dir = sampleDirection(roughness.r, param->ior, wi, normal, sampler);

        auto l = aten::length(result->dir);
        if (l == real(0)) {
            result->pdf = real(0);
            return;
        }

        result->pdf = pdf(roughness.r, param->ior, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API void MicrofacetRefraction::sample(
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
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec3(param->roughness));

        result->dir = sampleDirection(roughness.r, param->ior, wi, normal, sampler);
        result->pdf = pdf(roughness.r, param->ior, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API MaterialSampling MicrofacetRefraction::sample(
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

    bool MicrofacetRefraction::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1 || b2;
    }
}
