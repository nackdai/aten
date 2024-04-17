#include "material/ggx.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // Microfacet Models for Refraction through Rough Surfaces
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    // NOTE
    // http://qiita.com/_Pheema_/items/f1ffb2e38cc766e6e668

    AT_DEVICE_API float MicrofacetGGX::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec4(param->standard.roughness));

        auto ret = ComputeProbabilityToSampleOutputVector(roughness.r, n, wi, wo);
        return ret;
    }

    AT_DEVICE_API aten::vec3 MicrofacetGGX::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec4(param->standard.roughness));

        aten::vec3 dir = SampleDirection(roughness.r, normal, wi, sampler);

        return dir;
    }

    AT_DEVICE_API aten::vec3 MicrofacetGGX::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec4(param->standard.roughness));

        float ior = param->standard.ior;

        aten::vec3 ret = ComputeBRDF(roughness.r, ior, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_API void MicrofacetGGX::sample(
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

        result->dir = SampleDirection(roughness.r, wi, normal, sampler);
        result->pdf = ComputeProbabilityToSampleOutputVector(roughness.r, normal, wi, result->dir);

        float ior = param->standard.ior;

        result->bsdf = ComputeBRDF(roughness.r, ior, normal, wi, result->dir);
    }

    AT_DEVICE_API float MicrofacetGGX::ComputeDistribution(
        const aten::vec3& m,
        const aten::vec3& n,
        float roughness)
    {
        const auto a = roughness;
        const auto a2 = a * a;

        const auto costheta = aten::abs(dot(m, n));
        const auto cos2 = costheta * costheta;

        const auto denom = (a2 - 1) * cos2 + 1.0f;
        const auto denom2 = denom * denom;

        const auto D = denom > 0 ? a2 / (AT_MATH_PI * denom2) : 0;

        return D;
    }

    /**
     * @brief Compute G2 shadowing masking function.
     * @note Compute height correlated Smith shadowing-masking.
     * @param[in] roughness Roughness parameter.
     * @param[in] view Vector to eye.
     * @param[in] light Vector to light.
     * @param[in] n Macrosurface normal.
     * @return G2 shadowing masking value.
     */
    AT_DEVICE_API float MicrofacetGGX::ComputeG2Smith(
        float roughness,
        const aten::vec3& view,
        const aten::vec3& light,
        const aten::vec3& n)
    {
        const auto lambda_wi = Lambda(roughness, view, n);
        const auto lambda_wo = Lambda(roughness, light, n);
        const auto g2 = 1.0f / (1.0f + lambda_wi + lambda_wo);
        return g2;
    }

    /**
     * @brief Compute lambda function for shadowing-masking function.
     * @param[in] roughness Roughness parameter.
     * @param[in] w Target vector.
     * @param[in] m Microsurface normal.
     * @param Lambda value for shadowing-masking function.
     */
    AT_DEVICE_API float MicrofacetGGX::Lambda(
        float roughness,
        const aten::vec3& w,
        const aten::vec3& n)
    {
        // NOTE:
        // https://jcgt.org/published/0003/02/03/paper.pdf
        // 5.3.
        const auto alpha = roughness;

        const auto cos_theta = aten::abs(dot(w, n));
        const auto cos2 = cos_theta * cos_theta;

        const auto sin2 = 1.0f - cos2;
        const auto tan2 = sin2 / cos2;

        const auto a2 = 1.0f / (alpha * alpha * tan2);

        const auto lambda = (-1.0f + aten::sqrt(1.0f + 1.0f / a2)) / 2.0f;

        return lambda;
    }

    /**
     * @brief Compute probability to sample specified output vector.
     * @param[in] roughness Roughness parameter.
     * @param[in] n Macrosurface normal.
     * @param[in] wi Incident vector.
     * @param[in] wo Output vector.
     * @return Probability to sample output vector.
     */
    AT_DEVICE_API float  MicrofacetGGX::ComputeProbabilityToSampleOutputVector(
        const float roughness,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        auto wh = normalize(-wi + wo);

        auto D = ComputeDistribution(wh, n, roughness);

        auto costheta = aten::abs(dot(wh, n));

        // For Jacobian |dwh/dwo|
        auto denom = 4 * aten::abs(dot(wo, wh));

        auto pdf = denom > 0 ? (D * costheta) / denom : 0;

        return pdf;
    }

    /**
     * @brief Sample direction for reflection.
     * @param[in] roughness Roughness parameter.
     * @param[in] wi Incident vector.
     * @param[in] n Macrosurface normal.
     * @param[in, out] sampler Sampler to sample
     * @return Reflect vector.
     */
    AT_DEVICE_API aten::vec3  MicrofacetGGX::SampleDirection(
        const float roughness,
        const aten::vec3& wi,
        const aten::vec3& n,
        aten::sampler* sampler)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();

        const auto m = SampleMicrosurfaceNormal(roughness, n, r1, r2);

        // We can assume ideal reflection on each micro surface.
        // So, compute ideal reflection vector based on micro surface normal.
        const auto wo = material::ComputeReflectVector(wi, m);

        return wo;
    }

    /**
     * @brief Sample microsurface normal.
     * @param[in] roughness Roughness parameter.
     * @param[in] n Macrosurface normal.
     * @param[in] r1 Rondam value by uniforma sampleing.
     * @param[in] r2 Rondam value by uniforma sampleing.
     * @return Microsurface normal.
     */
    AT_DEVICE_API aten::vec3  MicrofacetGGX::SampleMicrosurfaceNormal(
        const float roughness,
        const aten::vec3& n,
        float r1, float r2)
    {
        const auto a = roughness;

        auto theta = aten::atan(a * aten::sqrt(r1 / (1 - r1)));
        theta = ((theta >= 0) ? theta : (theta + 2 * AT_MATH_PI));

        const auto phi = 2 * AT_MATH_PI * r2;

        const auto costheta = aten::cos(theta);
        const auto sintheta = aten::sqrt(1 - costheta * costheta);

        const auto cosphi = aten::cos(phi);
        const auto sinphi = aten::sqrt(1 - cosphi * cosphi);

        // Ortho normal base.
        const auto t = aten::getOrthoVector(n);
        const auto b = normalize(cross(n, t));

        auto m = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
        m = normalize(m);

        return m;
    }

    /**
     * @brief Compute BRDF.
     * @param[in] albedo Albedo color.
     * @param[in] roughness Roughness parameter.
     * @param[in] ior Refraction index.
     * @param[in] wi Incident vector.
     * @param[in] wo Output vector.
     * @param[in] n Macrosurface normal.
     * @return BRDF.
     */
    AT_DEVICE_API aten::vec3 MicrofacetGGX::ComputeBRDF(
        const float roughness,
        const float ior,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto V = -wi;
        const auto L = wo;
        const auto N = n;

        // We can assume ideal reflection on each micro surface.
        // It means wo (= L) is computed as ideal reflection vector.
        // Then, we can compute micro surface normal as the half vector between incident and output vector.
        const auto H = normalize(L + V);

        auto NL = aten::abs(dot(N, L));
        auto NV = aten::abs(dot(N, V));

        // Assume index of refraction of the medie on the incident side is vacuum.
        const auto ni = 1.0F;
        const auto nt = ior;

        const auto D = ComputeDistribution(H, N, roughness);
        const auto G2 = ComputeG2Smith(roughness, V, L, N);
        const auto F = material::ComputeSchlickFresnel(ni, nt, L, H);

        const auto denom = 4 * NL * NV;

        const auto bsdf = denom > AT_MATH_EPSILON ? F * G2 * D / denom : 0.0f;

        return aten::vec3(bsdf);
    }

    bool MicrofacetGGX::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0.01F, 1.0F);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, 0.01F, 10.0F);
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1 || b2;
    }
}
