#include "material/oren_nayar.h"
#include "material/diffuse.h"
#include "material/sample_texture.h"

//#pragma optimize( "", off)

namespace AT_NAME {
    // NOTE
    // https://ja.wikipedia.org/wiki/%E3%82%AA%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%BB%E3%83%8D%E3%82%A4%E3%83%A4%E3%83%BC%E5%8F%8D%E5%B0%84
    // https://github.com/imageworks/OpenShadingLanguage/blob/master/src/testrender/shading.cpp

    AT_DEVICE_API float OrenNayar::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto NL = dot(normal, wo);
        auto NV = dot(normal, -wi);

        float pdf = 0;

        if (NL > 0) {
            pdf = NL / AT_MATH_PI;
        }

        return pdf;
    }

    AT_DEVICE_API aten::vec3 OrenNayar::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        auto dir = Diffuse::sampleDirection(normal, sampler);
        return dir;
    }

    inline AT_DEVICE_API aten::vec3 computeBsdf(
        float roughness,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto NL = dot(normal, wo);
        const auto NV = dot(normal, -wi);

#if 0
        // NOTE
        // https://ja.wikipedia.org/wiki/%E3%82%AA%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%BB%E3%83%8D%E3%82%A4%E3%83%A4%E3%83%BC%E5%8F%8D%E5%B0%84

        float cosThetaO = NL;
        float cosThetaI = NV;

        float sinThetaO = float(1) - cosThetaO * cosThetaO;
        float sinThetaI = float(1) - cosThetaI * cosThetaI;

        float tanThetaO = sinThetaO / std::abs(cosThetaO);
        float tanThetaI = sinThetaI / std::abs(cosThetaI);

        float sinAlpha = aten::cmpMax(sinThetaO, sinThetaI);
        float tanBeta = aten::cmpMin(tanThetaO, tanThetaI);

        // NOTE
        // cos(φi - φr) = cosφi * cosφr + sinφi * sinφr

        // NOTE
        // NTB座標系において、v = (x, y, z) = (sinφ、cosφ、cosθ).

        auto n = normal;

        aten::vec3 t, b;
        aten::tie(t, b) = aten::GetTangentCoordinate(n);

        auto localV = (-wi.x * t + -wi.y * b + -wi.z * n);

        float cosAzimuth = (wo.x * localV.x + wo.y * localV.y);

        if (!aten::isValid(cosAzimuth)) {
            cosAzimuth = float(0);
        }

        const float a = roughness;
        const float a2 = a * a;
        const float A = float(1) - float(0.5) * (a2 / (a2 + float(0.33)));
        const float B = float(0.45) * (a2 / (a2 + float(0.09)));

        auto bsdf = (1.0F / AT_MATH_PI) * (A + B * aten::cmpMax(float(0), cosAzimuth) * sinAlpha * tanBeta);
#else
        // NOTE
        // A tiny improvement of Oren-Nayar reflectance model
        // http://mimosa-pudica.net/improved-oren-nayar.html

        const float a = roughness;
        const float a2 = a * a;

        const float A = float(1) - float(0.5) * (a2 / (a2 + float(0.33)));
        const float B = float(0.45) * (a2 / (a2 + float(0.09)));

        const auto LV = dot(wo, -wi);

        const auto s = LV - NL * NV;
        const auto t = s <= 0 ? float(1) : s / aten::cmpMax(NL, NV);

        auto bsdf = (1.0F / AT_MATH_PI) * (A + B * aten::cmpMax(float(0), s / t));
#endif

        return aten::vec3(bsdf);
    }

    AT_DEVICE_API aten::vec3 OrenNayar::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->standard.roughness));

        auto bsdf = computeBsdf(
            roughness.r,
            normal,
            wi,
            wo,
            u, v);

        return bsdf;
    }

    AT_DEVICE_API void OrenNayar::sample(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        aten::sampler* sampler,
        float u, float v)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler);
        result->pdf = pdf(param, normal, wi, result->dir, u, v);
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
    }

    bool OrenNayar::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param.standard, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1;
    }
}
