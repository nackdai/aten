#include "material/retroreflective.h"
#include "material/sample_texture.h"
#include "material/beckman.h"
#include "material/lambert.h"

namespace AT_NAME
{
    // TODO
    // API name...
    static AT_DEVICE_MTRL_API real computeFresnelEx(
        real ni, real nt,
        const aten::vec3& normal,
        const aten::vec3& wi)
    {
        const auto cosi = dot(normal, wi);

        const auto nnt = ni / nt;
        const auto sini2 = real(1.0) - cosi * cosi;
        const auto sint2 = nnt * nnt * sini2;
        const auto cost = aten::sqrt(aten::cmpMax(real(0.0), real(1.0) - sint2));

        const auto rp = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
        const auto rs = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);

        const auto Rsp = (rp * rp + rs * rs) * real(0.5);
        return Rsp;
    }

    AT_DEVICE_MTRL_API real Retroreflective::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 N = normal;

        // レイが入射してくる側の物体の屈折率.
        const real ni = real(1);      // 真空
        const real nt = param->ior;   // 物体内部の屈折率.

        const auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->roughness)).r;

        // Compute the vector into the prismatic sheet.
        // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
        const auto nnt = ni / nt;
        const auto d = dot(V, N);
        auto refract = -nnt * (V - d * N) - aten::sqrt(real(1) - nnt * nnt * (1 - d * d)) * N;
        refract = normalize(refract);

        const auto era = getEffectiveRetroreflectiveArea(refract, N);

        const auto fresnel = computeFresnelEx(ni, nt, N, wi);

        const auto beckmanPdf = MicrofacetBeckman::pdf(param, normal, wi, wo, u, v);
        const auto diffusePdf = lambert::pdf(N, wo);

        auto retrofractivePdf = real(1.0);
        {
            const aten::vec3 L = wo;
            const aten::vec3 V_dash = 2 * dot(N, V) * N - V;

            // back vector
            const aten::vec3 B = normalize(L + V_dash);

            const auto costheta = aten::abs(dot(B, N));

            const auto D = MicrofacetBeckman::sampleBeckman_D(B, N, roughness);

            const auto denom = 4 * aten::abs(dot(L, B));

            retrofractivePdf = denom > 0 ? (D * costheta) / denom : 0;
        }

        const auto pdf = era * (fresnel * beckmanPdf + (real(1) - fresnel) * retrofractivePdf) + (real(1) - era) * diffusePdf;

        return pdf;
    }

    AT_DEVICE_MTRL_API real Retroreflective::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 Retroreflective::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 N = normal;

        // レイが入射してくる側の物体の屈折率.
        const real ni = real(1);      // 真空
        const real nt = param->ior;   // 物体内部の屈折率.

        const auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->roughness)).r;

        // Compute the vector into the prismatic sheet.
        // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
        const auto nnt = ni / nt;
        const auto d = dot(V, N);
        auto refract = -nnt * (V - d * N) - aten::sqrt(real(1) - nnt * nnt * (1 - d * d)) * N;
        refract = normalize(refract);

        const auto era = getEffectiveRetroreflectiveArea(refract, N);

        const auto sample = sampler->nextSample2D();
        auto r0 = sample.x;
        auto r1 = sample.y;

        aten::vec3 dir;

        if (r0 < era) {
            const auto fresnel = computeFresnelEx(ni, nt, N, wi);

            r0 /= era;

            if (r1 < fresnel) {
                // Beckman
                r1 /= fresnel;
            }
            else {
                // Retroreflective
                r1 -= fresnel;
                r1 /= (real(1) - fresnel);
            }

            dir = MicrofacetBeckman::sampleDirection(roughness, wi, N, r0, r1);
        }
        else {
            // Diffuse
            r0 -= era;
            r0 /= (real(1) - era);

            dir = lambert::sampleDirection(N, r0, r1);
        }

        return dir;
    }

    AT_DEVICE_MTRL_API aten::vec3 Retroreflective::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        const aten::vec3& in = ray.dir;

        return sampleDirection(&m_param, normal, in, u, v, sampler);
    }

    AT_DEVICE_MTRL_API aten::vec3 Retroreflective::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        return aten::vec3();
    }

    AT_DEVICE_MTRL_API real Retroreflective::getEffectiveRetroreflectiveArea(
        const aten::vec3& into_prismatic_sheet_dir,
        const aten::vec3& normal)
    {
        // NOTE
        // MeasureEffectiveRetroreflectiveArea/main.cpp
        constexpr real Step = AT_MATH_PI_HALF / 40;

        constexpr size_t TableSize = 25;
        constexpr std::array<std::array<real, 2>, TableSize>  EffectiveRetroreflectiveAreaTable = { {
            {real(0.000), real(0.672)},
            {real(2.250), real(0.669)},
            {real(4.500), real(0.662)},
            {real(6.750), real(0.651)},
            {real(9.000), real(0.634)},
            {real(11.250), real(0.612)},
            {real(13.500), real(0.589)},
            {real(15.750), real(0.559)},
            {real(18.000), real(0.526)},
            {real(20.250), real(0.484)},
            {real(22.500), real(0.438)},
            {real(24.750), real(0.389)},
            {real(27.000), real(0.336)},
            {real(29.250), real(0.281)},
            {real(31.500), real(0.223)},
            {real(33.750), real(0.161)},
            {real(36.000), real(0.128)},
            {real(38.250), real(0.109)},
            {real(40.500), real(0.092)},
            {real(42.750), real(0.072)},
            {real(45.000), real(0.047)},
            {real(47.250), real(0.034)},
            {real(49.500), real(0.018)},
            {real(51.750), real(0.008)},
            {real(54.000), real(0.001)},
        } };

        // Inverse normal to align the vector into the prismatic sheet.
        const auto c = dot(into_prismatic_sheet_dir, -normal);
        if (c < real(0.0)) {
            return real(0.0);
        }

        const auto theta = aten::acos(c);

        const auto idx = static_cast<size_t>(theta / Step);

        real a = real(0.0);
        real b = real(0.0);
        real t = real(0.0);

        if (idx >= TableSize) {
            return real(0.0);
        }
        else if (idx == TableSize - 1) {
            auto d = EffectiveRetroreflectiveAreaTable[idx][0];
            t = (d - theta) / Step;

            a = EffectiveRetroreflectiveAreaTable[idx][1];
        }
        else {
            auto d = EffectiveRetroreflectiveAreaTable[idx][0];
            t = (d - theta) / Step;

            a = EffectiveRetroreflectiveAreaTable[idx][1];
            b = EffectiveRetroreflectiveAreaTable[idx + 1][1];
        }

        const auto result = a * (1 - t) + b * t;
        return result;
    }

    AT_DEVICE_MTRL_API aten::vec3 Retroreflective::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 L = wo;
        const aten::vec3 N = normal;
        const aten::vec3 H = normalize(L + V);

        const aten::vec3 V_dash = 2 * dot(N, V) * N - V;

        // back vector
        const aten::vec3 B = normalize(L + V_dash);

        const auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->roughness)).r;

        aten::vec3 albedo = param->baseColor;
        albedo *= externalAlbedo;

        real fresnel = 1;

        // レイが入射してくる側の物体の屈折率.
        const real ni = real(1);      // 真空
        const real nt = param->ior;   // 物体内部の屈折率.

        // Compute the vector into the prismatic sheet.
        // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
        const auto nnt = ni / nt;
        const auto d = dot(V, N);
        auto refract = -nnt * (V - d * N) - aten::sqrt(real(1) - nnt * nnt * (1 - d * d)) * N;
        refract = normalize(refract);

        const auto era = getEffectiveRetroreflectiveArea(refract, N);

        // Beckman
        const auto beckman = MicrofacetBeckman::bsdf(albedo, roughness, nt, fresnel, normal, wi, wo, u, v);

        // Retroreflective
        aten::vec3 retroreflective;
        {
            const auto D = MicrofacetBeckman::sampleBeckman_D(B, N, roughness);
            const auto G = MicrofacetBeckman::sampleBeckman_G(V, N, H, roughness) * MicrofacetBeckman::sampleBeckman_G(L, N, H, roughness);
            real F(1);
            {
                // http://d.hatena.ne.jp/hanecci/20130525/p3

                auto r0 = (ni - nt) / (ni + nt);
                r0 = r0 * r0;

                auto LdotB = aten::abs(dot(L, B));

                F = r0 + (1 - r0) * aten::pow((1 - LdotB), 5);
            }
            const auto denom = real(4) * dot(N, L) * dot(N, V);

            retroreflective = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : aten::vec3(0);
        }

        // TODO
        // Use sub color?

        // Diffuse
        const auto diffuse = static_cast<aten::vec3>(albedo) / AT_MATH_PI;

        const auto bsdf = era * (fresnel * beckman + (real(1) - fresnel) * retroreflective) + (real(1) - era) * diffuse;
        return bsdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 Retroreflective::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return bsdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API MaterialSampling Retroreflective::sample(
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

        return ret;
    }

    AT_DEVICE_MTRL_API void Retroreflective::sample(
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

    AT_DEVICE_MTRL_API void Retroreflective::sample(
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

    bool Retroreflective::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }
}
