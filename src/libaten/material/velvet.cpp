#include "material/velvet.h"
#include "math/math.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // Production Friendly Microfacet Sheen BRDF.
    // http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf

    AT_DEVICE_MTRL_API real MicrofacetVelvet::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto ret = pdf(param->roughness, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_MTRL_API real MicrofacetVelvet::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        aten::vec3 dir = sampleDirection(param->roughness, normal, wi, sampler);

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, param->roughness, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        aten::vec3 ret = bsdf(albedo, param->roughness, ior, fresnel, normal, wi, wo, u, v);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    static AT_DEVICE_MTRL_API inline real sampleVelvet_D(
        const aten::vec3& h,
        const aten::vec3& n,
        real roughness)
    {
        real cosTheta = dot(n, h);
        if (cosTheta < real(0)) {
            return real(0);
        }

        real inv_r = real(1) / roughness;
        real sinTheta = aten::sqrt(aten::cmpMax(1 - cosTheta * cosTheta, real(0)));

        real D = ((real(2) + inv_r) * aten::pow(sinTheta, inv_r)) / (AT_MATH_PI_2);

        return D;
    }

    static AT_DEVICE_MTRL_API inline real interpVelvetParam(int i, real a)
    {
        // NOTE
        // a = (1 - r)^2

        static const real p0[] = { real(25.3245), real(3.32435), real(0.16801), real(-1.27393), real(-4.85967) };
        static const real p1[] = { real(21.5473), real(3.82987), real(0.19823), real(-1.97760), real(-4.32054) };

        // NOTE
        // P = (1 − r)^2 * p0 + (1 − (1 − r)^2) * p1
        // => P = a * p0 + (1 - a) * p1

        real p = a * p0[i] + (1 - a) + p1[i];

        return p;
    }

    static AT_DEVICE_MTRL_API inline real computeVelvet_L(real x, real roughness)
    {
        real r = roughness;
        real powOneMinusR = aten::pow(real(1) - r, real(2));

        real a = interpVelvetParam(0, powOneMinusR);
        real b = interpVelvetParam(1, powOneMinusR);
        real c = interpVelvetParam(2, powOneMinusR);
        real d = interpVelvetParam(3, powOneMinusR);
        real e = interpVelvetParam(4, powOneMinusR);

        // NOTE
        // L(x) = a / (1 + b * x^c) + d * x + e

        real L = a / (1 + b * aten::pow(x, c)) + d * x + e;

        return L;
    }

    static AT_DEVICE_MTRL_API inline real computeVelvet_Lambda(real cosTheta, real roughness)
    {
        real r = roughness;

        if (cosTheta < real(0.5)) {
            return aten::exp(computeVelvet_L(cosTheta, r));
        }

        return aten::exp(2 * computeVelvet_L(0.5, r) - computeVelvet_L(1 - cosTheta, r));
    }

    static AT_DEVICE_MTRL_API inline real sampleVelvet_G(real cos_wi, real cos_wo, real r)
    {
        cos_wi = real(cos_wi <= real(0) ? 0 : 1);
        cos_wo = real(cos_wo <= real(0) ? 0 : 1);

        real G = cos_wi * cos_wo / (1 + computeVelvet_Lambda(cos_wi, r) + computeVelvet_Lambda(cos_wo, r));

        return G;
    }

    AT_DEVICE_MTRL_API real MicrofacetVelvet::pdf(
        real roughness,
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        // NOTE
        // "We found plain uniform sampling of the upper hemisphere to be more effective".
        // とのことで、importance sampling でなく、uniform sampling がいいらしい...

        aten::vec3 V = -wi;
        aten::vec3 N = normal;

        auto cosTheta = dot(V, N);

        if (cosTheta < real(0)) {
            return real(0);
        }

        return 1 / AT_MATH_PI_HALF;
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::sampleDirection(
        real roughness,
        const aten::vec3& in,
        const aten::vec3& normal,
        aten::sampler* sampler)
    {
        auto n = normal;
        auto t = aten::getOrthoVector(n);
        auto b = cross(n, t);

        // NOTE
        // "We found plain uniform sampling of the upper hemisphere to be more effective".
        // とのことで、importance sampling でなく、uniform sampling がいいらしい...

        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        // NOTE
        // r2[0, 1] => phi[0, 2pi]
        auto phi = AT_MATH_PI_2 * r2;

        // NOTE
        // r1[0, 1] => cos_theta[0, 1]
        auto cosTheta = r1;
        auto sinTheta = aten::sqrt(1 - cosTheta * cosTheta);

        auto x = aten::cos(phi) * sinTheta;
        auto y = aten::sin(phi) * sinTheta;
        auto z = cosTheta;

        aten::vec3 dir = normalize((t * x + b * y + n * z));

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 MicrofacetVelvet::bsdf(
        const aten::vec3& albedo,
        const real roughness,
        const real ior,
        real& fresnel,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        auto NdotL = aten::abs(dot(N, L));
        auto NdotV = aten::abs(dot(N, V));

        // Compute D.
        real D = sampleVelvet_D(H, N, roughness);

        // Compute G.
        real G = sampleVelvet_G(NdotL, NdotV, roughness);

        auto denom = 4 * NdotL * NdotV;

        auto bsdf = denom > AT_MATH_EPSILON ? albedo * G * D / denom : aten::vec3(0);
        
        fresnel = real(0);

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API void MicrofacetVelvet::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param->roughness, wi, normal, sampler);
        result->pdf = pdf(param->roughness, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, param->roughness, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API void MicrofacetVelvet::sample(
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
        result->dir = sampleDirection(param->roughness, wi, normal, sampler);
        result->pdf = pdf(param->roughness, normal, wi, result->dir);

        auto albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;
        real ior = param->ior;

        result->bsdf = bsdf(albedo, param->roughness, ior, fresnel, normal, wi, result->dir, u, v);
        result->fresnel = fresnel;
    }

    AT_DEVICE_MTRL_API MaterialSampling MicrofacetVelvet::sample(
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

    bool MicrofacetVelvet::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }
}
