#include "material/oren_nayar.h"
#include "material/lambert.h"
#include "material/sample_texture.h"

//#pragma optimize( "", off)

namespace AT_NAME {
    // NOTE
    // https://ja.wikipedia.org/wiki/%E3%82%AA%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%BB%E3%83%8D%E3%82%A4%E3%83%A4%E3%83%BC%E5%8F%8D%E5%B0%84
    // https://github.com/imageworks/OpenShadingLanguage/blob/master/src/testrender/shading.cpp

    AT_DEVICE_MTRL_API real OrenNayar::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto NL = dot(normal, wo);
        auto NV = dot(normal, -wi);

        real pdf = 0;

        if (NL > 0) {
            pdf = NL / AT_MATH_PI;
        }

        return pdf;
    }

    AT_DEVICE_MTRL_API real OrenNayar::pdf(
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 OrenNayar::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        auto dir = lambert::sampleDirection(normal, sampler);
        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 OrenNayar::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal, 
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    inline AT_DEVICE_MTRL_API aten::vec3 computeBsdf(
        real roughness,
        const aten::vec3& albedo,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        aten::vec3 bsdf = aten::vec3(0);

        const auto NL = dot(normal, wo);
        const auto NV = dot(normal, -wi);

#if 0
        // NOTE
        // https://ja.wikipedia.org/wiki/%E3%82%AA%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%BB%E3%83%8D%E3%82%A4%E3%83%A4%E3%83%BC%E5%8F%8D%E5%B0%84

        real cosThetaO = NL;
        real cosThetaI = NV;

        real sinThetaO = real(1) - cosThetaO * cosThetaO;
        real sinThetaI = real(1) - cosThetaI * cosThetaI;

        real tanThetaO = sinThetaO / std::abs(cosThetaO);
        real tanThetaI = sinThetaI / std::abs(cosThetaI);

        real sinAlpha = aten::cmpMax(sinThetaO, sinThetaI);
        real tanBeta = aten::cmpMin(tanThetaO, tanThetaI);

        // NOTE
        // cos(φi - φr) = cosφi * cosφr + sinφi * sinφr
        
        // NOTE
        // NTB座標系において、v = (x, y, z) = (sinφ、cosφ、cosθ).

        auto n = normal;
        auto t = aten::getOrthoVector(n);
        auto b = cross(n, t);
        auto localV = (-wi.x * t + -wi.y * b + -wi.z * n);

        real cosAzimuth = (wo.x * localV.x + wo.y * localV.y);

        if (!aten::isValid(cosAzimuth)) {
            cosAzimuth = real(0);
        }
        
        const real a = roughness;
        const real a2 = a * a;
        const real A = real(1) - real(0.5) * (a2 / (a2 + real(0.33)));
        const real B = real(0.45) * (a2 / (a2 + real(0.09)));

        bsdf = (albedo / AT_MATH_PI) * (A + B * aten::cmpMax(real(0), cosAzimuth) * sinAlpha * tanBeta);
#else
        // NOTE
        // A tiny improvement of Oren-Nayar reflectance model
        // http://mimosa-pudica.net/improved-oren-nayar.html

        const real a = roughness;
        const real a2 = a * a;

        const real A = real(1) - real(0.5) * (a2 / (a2 + real(0.33)));
        const real B = real(0.45) * (a2 / (a2 + real(0.09)));

        const auto LV = dot(wo, -wi);

        const auto s = LV - NL * NV;
        const auto t = s <= 0 ? real(1) : s / aten::cmpMax(NL, NV);

        bsdf = (albedo / AT_MATH_PI) * (A + B * aten::cmpMax(real(0), s / t));
#endif

        return bsdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 OrenNayar::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec3(param->roughness));

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        auto bsdf = computeBsdf(
            roughness.r,
            albedo,
            normal,
            wi,
            wo,
            u, v);
        
        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API aten::vec3 OrenNayar::bsdf(
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

        auto bsdf = computeBsdf(
            roughness.r,
            albedo,
            normal,
            wi,
            wo,
            u, v);

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API aten::vec3 OrenNayar::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    AT_DEVICE_MTRL_API void OrenNayar::sample(
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
        result->pdf = pdf(param, normal, wi, result->dir, u, v);
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
    }

    AT_DEVICE_MTRL_API void OrenNayar::sample(
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
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v, externalAlbedo);
    }

    AT_DEVICE_MTRL_API MaterialSampling OrenNayar::sample(
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

    bool OrenNayar::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1;
    }
}