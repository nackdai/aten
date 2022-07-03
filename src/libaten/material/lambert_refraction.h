#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class LambertRefraction : public material {
        friend class MaterialFactory;

    private:
        LambertRefraction(
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(1),
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Lambert_Refraction, MaterialAttributeTransmission, albedo, ior, nullptr, normalMap)
        {}

        LambertRefraction(aten::Values& val);

        virtual ~LambertRefraction() {}

    public:
        static AT_DEVICE_MTRL_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wo);

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& externalAlbedo);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false)
        {
            MaterialSampling ret;

            result->dir = sampleDirection(param, normal, wi, real(0), real(0), sampler);
            result->pdf = pdf(normal, result->dir);
            result->bsdf = bsdf(param, externalAlbedo);
        }

        virtual AT_DEVICE_MTRL_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final
        {
            auto ret = pdf(normal, wo);
            return ret;
        }

        virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler,
            real pre_sampled_r) const override final
        {
            return sampleDirection(&m_param, normal, ray.dir, u, v, sampler);
        }

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r) const override final
        {
            auto ret = bsdf(&m_param, u, v);
            return ret;
        }

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            bool isLightPath = false) const override final
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

        virtual AT_DEVICE_MTRL_API real computeFresnel(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor = 1) const override final
        {
            return computeFresnel(&m_param, normal, wi, wo, outsideIor);
        }

        static AT_DEVICE_MTRL_API real computeFresnel(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor)
        {
            return real(1);
        }

        virtual bool edit(aten::IMaterialParamEditor* editor) override final
        {
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

            auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, real(0.01), real(10));
            auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

            return b0 || b1;
        }
    };
}
