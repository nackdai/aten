#include "material/ggx.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    // NOTE
    // http://qiita.com/_Pheema_/items/f1ffb2e38cc766e6e668

    AT_DEVICE_MTRL_API real MicrofacetGGX::pdf(
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

        auto ret = pdf(roughness.r, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_MTRL_API real MicrofacetGGX::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::sampleDirection(
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

        aten::vec3 dir = sampleDirection(roughness.r, normal, wi, sampler);

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::bsdf(
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

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::bsdf(
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

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    AT_DEVICE_MTRL_API real MicrofacetGGX::sampleGGX_D(
        const aten::vec3& wh,    // half
        const aten::vec3& n,    // normal
        real roughness)
    {
        // NOTE
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

        // NOTE
        // ((a^2 - 1) * cos^2 + 1)^2
        // (-> a^2 = a2, cos^2 = cos2)
        // ((a2 - 1) * cos2 + 1)^2
        //  = (a2cos2 + 1 - cos2)^2 = (a2cos2 + sin2)^2
        // (-> sin = sqrt(1 - cos2), sin2 = 1 - cos2)
        //  = a4 * cos4 + 2 * a2 * cos2 * sin2 + sin4
        //  = cos4 * (a4 + 2 * a2 * (sin2 / cos2) + (sin4 / cos4))
        //  = cos4 * (a4 + 2 * a2 * tan2 + tan4)
        //  = cos4 * (a2 + tan2)^2
        //  = (cos2 * (a2 + tan2))^2
        // (tan = sin / cos -> tan2 = sin2 / cos2)
        //  = (a2 * cos2 + sin2)^2
        //  = (a2 * cos2 + (1 - cos2))^2
        //  = ((a2 - 1) * cos2 + 1)^2

        real a = roughness;
        auto a2 = a * a;

        auto costheta = aten::abs(dot(wh, n));
        auto cos2 = costheta * costheta;

        auto denom = aten::pow((a2 - 1) * cos2 + 1, 2);

        auto D = denom > 0 ? a2 / (AT_MATH_PI * denom) : 0;

        return D;
    }

    AT_DEVICE_MTRL_API real MicrofacetGGX::computeGGXSmithG1(real roughness, const aten::vec3& v, const aten::vec3& n)
    {
        // NOTE
        // http://computergraphics.stackexchange.com/questions/2489/correct-form-of-the-ggx-geometry-term
        // http://gregory-igehy.hatenadiary.com/entry/2015/02/26/154142

        real a = roughness;

        real costheta = aten::abs(dot(v, n));

        real sintheta = aten::sqrt(1 - aten::clamp<real>(costheta * costheta, 0, 1));
        real tan = costheta > 0 ? sintheta / costheta : 0;

        real denom = aten::sqrt(1 + a * a * tan * tan);

        real ret = 2 / (1 + denom);

        return ret;
    }

    AT_DEVICE_MTRL_API real MicrofacetGGX::pdf(
        real roughness,
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        // NOTE
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

        auto wh = normalize(-wi + wo);

        auto costheta = aten::abs(dot(wh, normal));

        auto D = sampleGGX_D(wh, normal, roughness);

        auto denom = 4 * aten::abs(dot(wo, wh));

        auto pdf = denom > 0 ? (D * costheta) / denom : 0;

        return pdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::sampleDirection(
        real roughness,
        const aten::vec3& in,
        const aten::vec3& normal,
        aten::sampler* sampler)
    {
        const auto w = sampleNormal(roughness, normal, sampler);

        auto dir = in - 2 * dot(in, w) * w;

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::sampleNormal(
        const real roughness,
        const aten::vec3& normal,
        aten::sampler* sampler)
    {
        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        auto a = roughness;

        auto theta = aten::atan(a * aten::sqrt(r1 / (1 - r1)));
        theta = ((theta >= 0) ? theta : (theta + 2 * AT_MATH_PI));

        auto phi = 2 * AT_MATH_PI * r2;

        auto costheta = aten::cos(theta);
        auto sintheta = aten::sqrt(1 - costheta * costheta);

        auto cosphi = aten::cos(phi);
        auto sinphi = aten::sqrt(1 - cosphi * cosphi);

        // Ortho normal base.
        auto n = normal;
        auto t = aten::getOrthoVector(normal);
        auto b = normalize(cross(n, t));

        auto w = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
        w = normalize(w);

        return std::move(w);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetGGX::bsdf(
        const aten::vec3& albedo,
        const real roughness,
        const real ior,
        real& fresnel,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        // ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
        real ni = real(1);    // ^‹ó

        real nt = ior;        // •¨‘Ì“à•”‚Ì‹üÜ—¦.

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

        // Compute D.
        real D = sampleGGX_D(H, N, roughness);

        // Compute G.
        real G(1);
        {
            auto G1_lh = computeGGXSmithG1(roughness, L, N);
            auto G1_vh = computeGGXSmithG1(roughness, V, N);

            G = G1_lh * G1_vh;
        }

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

        auto bsdf = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : aten::vec3(0);
        
        fresnel = F;

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API void MicrofacetGGX::sample(
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

        result->dir = sampleDirection(roughness.r, wi, normal, sampler);
        result->pdf = pdf(roughness.r, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API void MicrofacetGGX::sample(
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

        result->dir = sampleDirection(roughness.r, wi, normal, sampler);
        result->pdf = pdf(roughness.r, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API MaterialSampling MicrofacetGGX::sample(
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

    bool MicrofacetGGX::edit(aten::IMaterialParamEditor* editor)
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
