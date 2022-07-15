#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class CarPaint : public material {
        friend class MaterialFactory;

    private:
        CarPaint(
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        CarPaint(
            aten::vec3 baseColor,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet)
        {
            m_param.baseColor = baseColor;
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        CarPaint(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, param.baseColor, 1, albedoMap, normalMap)
        {
            m_param = param;
            m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
        }

        CarPaint(aten::Values& val);

        virtual ~CarPaint() {}

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
            aten::sampler* sampler,
            real pre_sampled_r);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            const aten::vec3& externalAlbedo,
            real pre_sampled_r);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            bool isLightPath = false);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false);

        virtual AT_DEVICE_MTRL_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler,
            real pre_sampled_r) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r) const override final;

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            bool isLightPath = false) const override final;

        virtual AT_DEVICE_MTRL_API real applyNormalMap(
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            real u, real v,
            const aten::vec3& wi,
            aten::sampler* sampler) const override final;

        static AT_DEVICE_MTRL_API real applyNormalMap(
            const aten::MaterialParameter* param,
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            real u, real v,
            const aten::vec3& wi,
            aten::sampler* sampler);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
