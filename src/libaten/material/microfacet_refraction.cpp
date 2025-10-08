#include "material/microfacet_refraction.h"
#include "material/ggx.h"
#include "material/refraction.h"
#include "material/sample_texture.h"
#include "material/specular.h"

#include "misc/misc.h"

#pragma optimize( "", off)

namespace AT_NAME
{
    AT_DEVICE_API float MicrofacetRefraction::pdf(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        const float u, const float v)
    {
        AT_ASSERT(false);
        return 1.0F;
    }

    AT_DEVICE_API aten::vec3 MicrofacetRefraction::sampleDirection(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const float u, const float v,
        aten::sampler* sampler)
    {
        AT_ASSERT(false);
        return aten::vec3();
    }

    AT_DEVICE_API aten::vec3 MicrofacetRefraction::bsdf(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        const float u, const float v)
    {
        AT_ASSERT(false);
        return aten::vec3();
    }

    AT_DEVICE_API void MicrofacetRefraction::sample(
        AT_NAME::MaterialSampling& result,
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        aten::sampler* sampler,
        const float u, const float v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param.roughnessMap,
            u, v,
            aten::vec4(param.standard.roughness));

        SampleMicrofacetRefraction(
            result,
            roughness.r, param.standard.ior,
            n, wi,
            sampler);
    }

    AT_DEVICE_API void MicrofacetRefraction::SampleMicrofacetRefraction(
        AT_NAME::MaterialSampling& result,
        const float roughness,
        const float ior,
        const aten::vec3& n,
        const aten::vec3& wi,
        aten::sampler* sampler)
    {
        auto ni = 1.0F;
        auto nt = ior;

        const auto V = -wi;
        auto N = n;

        const auto is_enter = dot(V, N) >= 0.0F;

        if (!is_enter) {
            N = -n;
            aten::swap(ni, nt);
        }

        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();
        const auto m = MicrofacetGGX::SampleMicrosurfaceNormal(roughness, N, r1, r2);

        const auto F = material::ComputeSchlickFresnel(ni, nt, wi, m);

        const auto R = F;       // reflectance.
        const auto T = 1 - R;   // transmittance.

        const auto prob = R;

        const auto u = sampler->nextSample();

        if (u < prob) {
            // Reflection.
            const auto wo = material::ComputeReflectVector(wi, m);

            const auto VN = aten::dot(V, N);
            const auto LN = aten::dot(wo, N);

            // Incident vector and output vector have to be on the same hemisphere.
            // It means they have to be the same direction with the macro surface normal.
            if (VN * LN < 0) {
                // TODO:
                // Return invalid sampling accordingly.
                // Return zero brdf alternatively at this moment.
                // Currently, returning invalid direction is not exepcted.
                // So, return ideal reflection vector.
                result.dir = material::ComputeReflectVector(wi, N);
                result.pdf = 1.0f;
                result.bsdf = aten::vec3(0);
                return;
            }

            result.dir = wo;

            result.pdf = MicrofacetGGX::ComputePDFWithHalfVector(roughness, N, m, wo);
            result.pdf *= prob;

            result.bsdf = MicrofacetGGX::ComputeBRDF(roughness, ior, N, wi, wo);
        }
        else {
            // Refraction.
            const auto wo = material::ComputeRefractVector(ni, nt, wi, m);

            const auto D = MicrofacetGGX::ComputeDistribution(m, N, roughness);
            const auto G = MicrofacetGGX::ComputeG2Smith(roughness, V, wo, N);

            const auto LH = aten::abs(dot(wo, m));
            const auto VH = aten::abs(dot(V, m));

            const auto denom = ni * dot(V, m) + nt * dot(wo, m);
            const auto denom2 = denom * denom;

            const auto costheta = aten::abs(dot(m, n));
            const auto nt2 = nt * nt;

            result.pdf = denom2 > 0 ? D * costheta * (nt2 * LH / denom2) : 1.0F;
            result.pdf *= 1.0F - prob;

            result.dir = wo;

            auto VN = dot(V, N);
            auto LN = dot(wo, N);

            // Incident vector and output vector "don't" have to be on the same hemisphere.
            // It means they "don't" have to be the same direction with the macro surface normal.
            if (VN * LN > 0) {
                // TODO:
                // Return invalid sampling accordingly.
                // Return zero bsdf alternatively at this moment.
                // Currently, returning invalid direction is not exepcted.
                // So, return ideal refraction vector.
                result.dir = material::ComputeRefractVector(ni, nt, wi, N);
                result.pdf = 1.0f;
                result.bsdf = aten::vec3(0);
                return;
            }

            VN = aten::abs(VN);
            LN = aten::abs(LN);

            const auto bsdf = denom2 > 0 ? ((VH * LH) / (VN * LN)) * (nt2 * T * D * G / denom2) : 0.0F;
            result.bsdf = aten::vec3(bsdf);
        }
    }

    bool MicrofacetRefraction::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = editor->edit("roughness", param_.standard.roughness, 0.001F, 1.0F);
        auto b1 = editor->edit("ior", param_.standard.ior, 0.01F, 10.0F);
        auto b2 = editor->edit("baseColor", param_.baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, roughnessMap);

        return b0 || b1 || b2;
    }
}
