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

        aten::vec3 dir = SampleReflectDirection(roughness.r, normal, wi, sampler);

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

        result->dir = SampleReflectDirection(roughness.r, wi, normal, sampler);
        result->pdf = ComputeProbabilityToSampleOutputVector(roughness.r, normal, wi, result->dir);

        float ior = param->standard.ior;

        result->bsdf = ComputeBRDF(roughness.r, ior, normal, wi, result->dir);
    }

    bool MicrofacetGGX::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param.standard, roughness);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, float(0.01), float(10));
        auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

        return b0 || b1 || b2;
    }
}
