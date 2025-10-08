#include "material/beckman.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // Microfacet Models for Refraction through Rough Surfaces
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    AT_DEVICE_API float MicrofacetBeckman::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->standard.roughness));
        auto ret = ComputePDF(roughness.r, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->standard.roughness));
        aten::vec3 dir = sampleDirection(roughness.r, wi, normal, sampler);
        return dir;
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(param->roughnessMap, u, v, aten::vec4(param->standard.roughness));

        float ior = param->standard.ior;

        aten::vec3 ret = ComputeBRDF(roughness.r, ior, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::sampleDirection(
        const float roughness,
        const aten::vec3& in,
        const aten::vec3& normal,
        aten::sampler* sampler)
    {
        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        return SampleDirection(roughness, in, normal, r1, r2);
    }

    AT_DEVICE_API void MicrofacetBeckman::sample(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec4(param->standard.roughness));

        result->dir = sampleDirection(roughness.r, wi, normal, sampler);
        result->pdf = ComputePDF(roughness.r, normal, wi, result->dir);

        float ior = param->standard.ior;

        result->bsdf = ComputeBRDF(roughness.r, ior, normal, wi, result->dir);
    }

    bool MicrofacetBeckman::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, param_.standard, roughness, 0.01F, 1.0F);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, param_.standard, ior, 0.01F, 10.0F);
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, param_, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, roughnessMap);

        return b0 || b1 || b2;
    }

    AT_DEVICE_API float MicrofacetBeckman::ComputeDistribution(
        const aten::vec3& m,
        const aten::vec3& n,
        float roughness)
    {
        const auto costheta = aten::abs(dot(m, n));
        if (costheta <= 0) {
            return 0;
        }

        const auto theta = aten::acos(costheta);

        const auto cos2 = costheta * costheta;
        const auto cos4 = cos2 * cos2;

        const auto sintheta = aten::sqrt(1 - cos2);
        const auto tantheta = sintheta / costheta;
        const auto tan2 = tantheta * tantheta;

        const auto a = roughness;
        const auto a2 = a * a;

        auto D = 1.0f / (AT_MATH_PI * a2 * cos4);
        D *= aten::exp(-tan2 / a2);

        return D;
    }

    AT_DEVICE_API float MicrofacetBeckman::ComputePDF(
        const float roughness,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto wh = normalize(-wi + wo);

        const auto costheta = aten::abs(dot(wh, n));

        auto D = ComputeDistribution(wh, n, roughness);

        // For Jacobian |dwh/dwo|
        const auto denom = 4 * aten::abs(dot(wo, wh));

        const auto pdf = denom > 0 ? (D * costheta) / denom : 0;

        return pdf;
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::SampleMicrosurfaceNormal(
        const float roughness,
        const aten::vec3& n,
        float r1, float r2)
    {
        const auto a = roughness;
        const auto a2 = a * a;

        // NOTE:
        // log can't accept zero. If r1 is 1, (1 - r1) is zero.
        // To avoid it, if r1 is 1, to lessen r1 less than 1, multipley 0.99.
        const auto theta = aten::atan(aten::sqrt(-a2 * aten::log(1.0F - r1 * 0.99F)));

        const auto phi = AT_MATH_PI_2 * r2;

        const auto costheta = aten::cos(theta);
        const auto sintheta = aten::sin(theta);

        const auto cosphi = aten::cos(phi);
        const auto sinphi = aten::sin(phi);

        // Ortho normal base.
        aten::vec3 t, b;
        aten::tie(t, b) = aten::GetTangentCoordinate(n);

        auto m = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
        m = normalize(m);

        return m;
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::SampleDirection(
        const float roughness,
        const aten::vec3& wi,
        const aten::vec3& n,
        float r1, float r2)
    {
        const auto m = SampleMicrosurfaceNormal(roughness, n, r1, r2);

        // We can assume ideal reflection on each micro surface.
        // So, compute ideal reflection vector based on micro surface normal.
        const auto wo = material::ComputeReflectVector(wi, m);

        return wo;
    }

    AT_DEVICE_API float MicrofacetBeckman::ComputeG1(
        float roughness,
        const aten::vec3& v,
        const aten::vec3& n)
    {
        const auto costheta = aten::saturate(aten::abs(dot(v, n)));
        const auto sintheta = aten::sqrt(1.0F - costheta * costheta);
        const auto tantheta = sintheta / costheta;

        const auto a = 1.0F / (roughness * tantheta);
        const auto a2 = a * a;

        if (a < 1.6F) {
            return (3.535F * a + 2.181F * a2) / (1.0F + 2.276F * a + 2.577F * a2);
        }
        else {
            return 1.0F;
        }
    }

    AT_DEVICE_API aten::vec3 MicrofacetBeckman::ComputeBRDF(
        const float roughness,
        const float ior,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float* fresnel/*= nullptr*/)
    {
        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = n;

        // We can assume ideal reflection on each micro surface.
        // It means wo (= L) is computed as ideal reflection vector.
        // Then, we can compute micro surface normal as the half vector between incident and output vector.
        aten::vec3 H = normalize(L + V);

        auto NH = aten::abs(dot(N, H));
        auto VH = aten::abs(dot(V, H));
        auto NL = aten::abs(dot(N, L));
        auto NV = aten::abs(dot(N, V));

        // Assume index of refraction of the medie on the incident side is vacuum.
        const auto ni = 1.0F;
        const auto nt = ior;

        const auto D = ComputeDistribution(H, N, roughness);
        const auto G2 = ComputeG1(roughness, V, N) * ComputeG1(roughness, L, N);
        const auto F = material::ComputeSchlickFresnel(ni, nt, L, H);

        if (fresnel) {
            *fresnel = F;
        }

        const auto denom = 4 * NL * NV;

        const auto bsdf = denom > AT_MATH_EPSILON ? F * G2 * D / denom : 0.0f;

        return aten::vec3(bsdf);
    }
}
