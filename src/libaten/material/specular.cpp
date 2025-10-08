#include "material/specular.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    AT_DEVICE_API float specular::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        return ComputePDF();
    }

    AT_DEVICE_API aten::vec3 specular::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        return ComputeReflectVector(wi, normal);
    }

    AT_DEVICE_API aten::vec3 specular::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        return ComputeBRDF(wo, normal);
    }

    AT_DEVICE_API void specular::sample(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        float u, float v)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler);
        result->pdf = pdf(param, normal, wi, result->dir, u, v);
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
    }

    bool specular::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, param_.standard, ior, float(0.01), float(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, param_, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, param_, normalMap);

        return b0 || b1;
    }
}
