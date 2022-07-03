#include "material/specular.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    AT_DEVICE_MTRL_API real specular::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        return real(1);
    }

    AT_DEVICE_MTRL_API real specular::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 specular::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        auto reflect = wi - 2 * dot(normal, wi) * normal;
        reflect = normalize(reflect);

        return reflect;
    }

    AT_DEVICE_MTRL_API aten::vec3 specular::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler,
        real pre_sampled_r) const
    {
        const aten::vec3& in = ray.dir;

        return sampleDirection(&m_param, normal, in, u, v, sampler);
    }

    AT_DEVICE_MTRL_API aten::vec3 specular::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        // NOTE
        // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularReflection

        auto bsdf = param->baseColor;

        // For canceling cosine factor.
        auto c = dot(normal, wo);
        bsdf /= abs(c);

        bsdf *= sampleTexture(param->albedoMap, u, v, aten::vec4(real(1)));

#if 0
        if (param->ior > real(0)) {
            real nc = real(1);      // 真空の屈折率.
            real nt = param->ior;   // 物体内部の屈折率.

            // SchlickによるFresnelの反射係数の近似を使う.
            const real a = nt - nc;
            const real b = nt + nc;
            const real r0 = (a * a) / (b * b);

            const real c = dot(normal, wo);

            // 反射方向の光が反射してray.dirの方向に運ぶ割合。同時に屈折方向の光が反射する方向に運ぶ割合.
            const real fresnel = r0 + (1 - r0) * aten::pow(c, 5);

            bsdf *= fresnel;
        }
#endif

        return bsdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 specular::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        auto c = dot(normal, wo);

        aten::vec3 bsdf = param->baseColor;
        bsdf *= externalAlbedo;

        return bsdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 specular::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        real pre_sampled_r) const
    {
        return bsdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API MaterialSampling specular::sample(
        const aten::ray& ray,
        const aten::vec3& normal,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
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

    AT_DEVICE_MTRL_API void specular::sample(
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

    AT_DEVICE_MTRL_API void specular::sample(
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

    bool specular::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }
}
