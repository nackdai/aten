#include "material/velvet.h"
#include "material/diffuse.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    // NOTE
    // Production Friendly Microfacet Sheen BRDF.
    // https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_sheen.pdf

    AT_DEVICE_API float MicrofacetVelvet::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto ret = ComputePDF(n, wo);
        return ret;
    }

    AT_DEVICE_API aten::vec3 MicrofacetVelvet::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        auto dir = SampleDirection(normal, r1, r2);
        while (dot(normal, dir) >= 0) {
            r1 = sampler->nextSample();
            r2 = sampler->nextSample();
            dir = SampleDirection(normal, r1, r2);
        }

        return dir;
    }

    AT_DEVICE_API aten::vec3 MicrofacetVelvet::bsdf(
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

        const auto ret = ComputeBRDF(roughness.r, normal, wi, wo);
        return ret;
    }

    AT_DEVICE_API float MicrofacetVelvet::ComputeDistribution(
        const aten::vec3& m,
        const aten::vec3& n,
        const float roughness)
    {
        const auto cos_theta = aten::abs(dot(m, n));

        const auto inv_r = 1.0F / roughness;
        const auto sin_theta = aten::sqrt(aten::saturate(1 - cos_theta * cos_theta));

        const auto D = ((2.0F + inv_r) * aten::pow(sin_theta, inv_r)) / (AT_MATH_PI_2);

        return D;
    }

    AT_DEVICE_API inline float MicrofacetVelvet::InterpolateVelvetParam(const int32_t idx, float interp_factor)
    {
        // NOTE:
        // Interpolate factor : interp_factor = (1 - r)^2

        constexpr std::array p0 = { 25.3245F, 3.32435F, 0.16801F, -1.27393F, -4.85967F };
        constexpr std::array p1 = { 21.5473F, 3.82987F, 0.19823F, -1.97760F, -4.32054F };

        // NOTE:
        // P = (1 − r)^2 * p0 + (1 − (1 − r)^2) * p1
        // => P = interp_factor * p0 + (1 - interp_factor) * p1

        float p = interp_factor * p0[idx] + (1 - interp_factor) + p1[idx];

        return p;
    }

    AT_DEVICE_API float MicrofacetVelvet::ComputeVelvetLForLambda(const float x, const float roughness)
    {
        // NOTE:
        // Interpolate factor : interp_factor = (1 - r)^2
        const auto r = roughness;
        const auto interp_factor = aten::pow(float(1) - r, float(2));

        const auto a = InterpolateVelvetParam(0, interp_factor);
        const auto b = InterpolateVelvetParam(1, interp_factor);
        const auto c = InterpolateVelvetParam(2, interp_factor);
        const auto d = InterpolateVelvetParam(3, interp_factor);
        const auto e = InterpolateVelvetParam(4, interp_factor);

        // NOTE:
        // L(x) = a / (1 + b * x^c) + d * x + e

        const auto L = a / (1 + b * aten::pow(x, c)) + d * x + e;

        return L;
    }

    AT_DEVICE_API float MicrofacetVelvet::ComputeVelvetLambda(
        const float roughness,
        const aten::vec3& w,
        const aten::vec3& m)
    {
        const auto r = roughness;
        const auto cos_theta = aten::saturate(aten::abs(dot(w, m)));

        if (cos_theta < 0.5F) {
            return aten::exp(ComputeVelvetLForLambda(cos_theta, r));
        }

        return aten::exp(2.0F * ComputeVelvetLForLambda(0.5F, r) - ComputeVelvetLForLambda(1 - cos_theta, r));
    }

    AT_DEVICE_API float MicrofacetVelvet::ComputeShadowingMaskingFunction(
        const float roughness,
        const aten::vec3& view,
        const aten::vec3& light,
        const aten::vec3& n)
    {
        auto lambda_wi = ComputeVelvetLambda(roughness, view, n);
        const auto lambda_wo = ComputeVelvetLambda(roughness, light, n);

        const auto cos_theta_wi = aten::saturate(aten::abs(dot(view, n)));
        const auto cos_theta_wo = aten::saturate(aten::abs(dot(light, n)));

        // For Terminator Softening
        lambda_wi = aten::pow(lambda_wi, 1.0F + 2.0F * aten::pow(1.0F - cos_theta_wi, 8));

        float G = 1.0F / (1.0F + lambda_wi + lambda_wo);

        return G;
    }

    AT_DEVICE_API float MicrofacetVelvet::ComputePDF(
        const aten::vec3& n,
        const aten::vec3& wo)
    {
        // NOTE:
        // The papaer mentions "We found plain uniform sampling of the upper hemisphere to be more effective".
        // So, sample based on hemisphere uniform sampling not importance sampling.
        return Diffuse::ComputePDF(n, wo);
    }

    AT_DEVICE_API aten::vec3 MicrofacetVelvet::SampleDirection(
        const aten::vec3& n,
        float r1, float r2)
    {
        // NOTE:
        // The papaer mentions "We found plain uniform sampling of the upper hemisphere to be more effective".
        // So, sample based on hemisphere uniform sampling not importance sampling.
        return Diffuse::SampleDirection(n, r1, r2);
    }

    AT_DEVICE_API aten::vec3 MicrofacetVelvet::ComputeBRDF(
        const float roughness,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto V = -wi;
        const auto L = wo;
        const auto N = n;
        const auto H = normalize(L + V);

        auto NL = aten::abs(dot(N, L));
        auto NV = aten::abs(dot(N, V));

        const auto D = ComputeDistribution(H, N, roughness);
        const auto G = ComputeShadowingMaskingFunction(roughness, V, L, N);

        // NOTE:
        // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md
        // The Fresnel term may be omitted, i.e., F = 1.
        // https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2022x.md.html#components/sheen
        // Looks like fresnel term is omitted as well...
        constexpr auto F = 1.0F;

        const auto denom = 4 * NL * NV;

        const auto bsdf = denom > AT_MATH_EPSILON ? F * G * D / denom : 0.0F;

        return aten::vec3(bsdf);
    }

    AT_DEVICE_API void MicrofacetVelvet::sample(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        aten::sampler* sampler,
        float u, float v)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();

        result->dir = SampleDirection(normal, r1, r2);
        result->pdf = ComputePDF(normal, result->dir);

        const auto roughness = AT_NAME::sampleTexture(
            param->roughnessMap,
            u, v,
            aten::vec4(param->standard.roughness));

        result->bsdf = ComputeBRDF(roughness.r, normal, wi, result->dir);
    }

    bool MicrofacetVelvet::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, param_.standard, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, param_, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, normalMap);

        return b0 || b1;
    }
}
