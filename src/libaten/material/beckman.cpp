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

        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec4(float(1)));

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
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0.01F, 1.0F);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, 0.01F, 10.0F);
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1 || b2;
    }
}
