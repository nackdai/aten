#include "material/beckman.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    AT_DEVICE_MTRL_API real MicrofacetBeckman::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec3(param->roughness));
        auto ret = pdf(roughness.r, normal, wi, wo);
        return ret;
    }

    real MicrofacetBeckman::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        auto ret = pdf(&m_param, normal, wi, wo, u, v);
        return ret;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec3(param->roughness));
        aten::vec3 dir = sampleDirection(roughness.r, wi, normal, sampler);
        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        auto dir = sampleDirection(&m_param, normal, ray.dir, u, v, sampler);
        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec3(param->roughness));

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec3(param->roughness));

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        auto ret = bsdf(&m_param, normal, wi, wo, u, v);
        return std::move(ret);
    }

    static AT_DEVICE_MTRL_API real sampleBeckman_D(
        const aten::vec3& wh,    // half
        const aten::vec3& n,    // normal
        real roughness)
    {
        // NOTE
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

        auto costheta = dot(wh, n);

        if (costheta <= 0) {
            return 0;
        }

        auto cos2 = costheta * costheta;

        auto sintheta = aten::sqrt(1 - aten::clamp<real>(cos2, 0, 1));
        auto tantheta = sintheta / costheta;
        auto tan2 = tantheta * tantheta;

        real a = roughness;
        auto a2 = a * a;

        auto D = 1 / (AT_MATH_PI * a2 * cos2 * cos2);
        D *= aten::exp(-tan2 / a2);

        return D;
    }

    AT_DEVICE_MTRL_API real MicrofacetBeckman::pdf(
        const real roughness,
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        // NOTE
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

        auto wh = normalize(-wi + wo);

        auto costheta = aten::abs(dot(wh, normal));

        auto D = sampleBeckman_D(wh, normal, roughness);

        auto denom = 4 * aten::abs(dot(wo, wh));

        auto pdf = denom > 0 ? (D * costheta) / denom : 0;

        return pdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::sampleDirection(
        const real roughness,
        const aten::vec3& in,
        const aten::vec3& normal,
        aten::sampler* sampler)
    {
        // NOTE
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        auto a = roughness;
        auto a2 = a * a;

        auto theta = aten::sqrt(-a2 * aten::log(1 - r1 * real(0.99)));
        theta = aten::atan(theta);
        theta = ((theta >= real(0)) ? theta : (theta + 2 * AT_MATH_PI));

        auto phi = real(2) * AT_MATH_PI * r2;

        auto costheta = aten::cos(theta);
        auto sintheta = aten::sqrt(real(1) - costheta * costheta);

        auto cosphi = aten::cos(phi);
        auto sinphi = aten::sqrt(1 - cosphi * cosphi);

        // Ortho normal base.
        auto n = normal;
        auto t = aten::getOrthoVector(normal);
        auto b = normalize(cross(n, t));

        auto w = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
        w = normalize(w);

        auto dir = in - real(2) * dot(in, w) * w;

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetBeckman::bsdf(
        const aten::vec3& albedo,
        const real roughness,
        const real ior,
        real& fresnel,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        // ÉåÉCÇ™ì¸éÀÇµÇƒÇ≠ÇÈë§ÇÃï®ëÃÇÃã¸ê‹ó¶.
        real ni = real(1);    // ê^ãÛ

        real nt = ior;        // ï®ëÃì‡ïîÇÃã¸ê‹ó¶.

        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        // TODO
        // DesneyÇæÇ∆absÇµÇƒÇ»Ç¢Ç™ÅAAMDÇÃÇÕÇµÇƒÇ¢ÇÈ....
        auto NdotH = aten::abs(dot(N, H));
        auto VdotH = aten::abs(dot(V, H));
        auto NdotL = aten::abs(dot(N, L));
        auto NdotV = aten::abs(dot(N, V));

        auto a = roughness;

        // Compute D.
        real D = sampleBeckman_D(H, N, a);

        // Compute G.
        real G(1);
        {
            // NOTE
            // http://graphicrants.blogspot.jp/2013/08/specular-brdf-reference.html

            auto c = NdotV < real(1) ? NdotV / (a * aten::sqrt(real(1) - NdotV * NdotV)) : real(0);
            auto c2 = c * c;

            if (c < real(1.6)) {
                G = (real(3.535) * c + real(2.181) * c2) / (real(1) + real(2.276) * c + real(2.577) * c2);
            }
            else {
                G = real(1);
            }
        }

        real F(1);
        {
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) Å‡ R0 + (1 - R0)(1 - cosÉ¶)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (ni - nt) / (ni + nt);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(L, H));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

        auto denom = real(4) * NdotL * NdotV;

        auto bsdf = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : aten::vec3(0);

        fresnel = F;

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API void MicrofacetBeckman::sample(
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

        real fresnel = real(1);

        real ior = param->ior;

        auto albedo = param->baseColor;
        albedo *= sampleTexture(
            param->albedoMap,
            u, v,
            aten::vec3(real(1)));

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API void MicrofacetBeckman::sample(
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

        real fresnel = real(1);

        real ior = param->ior;

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API MaterialSampling MicrofacetBeckman::sample(
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

    bool MicrofacetBeckman::edit(aten::IMaterialParamEditor* editor)
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
