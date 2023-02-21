#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class LambertRefraction : public material {
        friend class material;

    private:
        LambertRefraction(
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(1),
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Lambert_Refraction, MaterialAttributeTransmission, albedo, ior)
        {
            setTextures(nullptr, normalMap, nullptr);
        }

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
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false);

        static AT_DEVICE_MTRL_API void sample(
            AT_NAME::MaterialSampling* result,
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
