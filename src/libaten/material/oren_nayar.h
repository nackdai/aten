#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class OrenNayar : public material {
        friend class material;

    private:
        OrenNayar(
            const aten::vec3& albedo = aten::vec3(0.5),
            real roughness = real(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::OrneNayar, MaterialAttributeLambert, albedo, 1)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
            m_param.standard.roughness = aten::clamp<real>(roughness, 0, 1);
        }

        OrenNayar(aten::Values& val);

        virtual ~OrenNayar() {}

    public:
        static AT_DEVICE_MTRL_API real pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            const aten::vec4& externalAlbedo);

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
            real u, real v,
            const aten::vec4& externalAlbedo,
            bool isLightPath = false);

        static AT_DEVICE_MTRL_API real computeFresnel(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor)
        {
            return real(1);
        }

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
