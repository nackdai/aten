#pragma once

#include "material/material.h"
#include "image/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class CarPaint : public material {
        friend class material;

    private:
        CarPaint(
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, aten::MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        CarPaint(
            aten::vec3 baseColor,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, aten::MaterialAttributeMicrofacet)
        {
            m_param.baseColor = baseColor;
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        CarPaint(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(param, aten::MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        CarPaint(aten::Values& val);

        virtual ~CarPaint() {}

    public:
        static AT_DEVICE_API float pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v,
            aten::sampler* sampler,
            float pre_sampled_r);

        static AT_DEVICE_API aten::vec3 bsdf(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v,
            float pre_sampled_r);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v,
            const aten::vec3& externalAlbedo,
            float pre_sampled_r);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            float pre_sampled_r,
            float u, float v,
            bool isLightPath = false);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            float pre_sampled_r,
            float u, float v,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false);

        static AT_DEVICE_API float applyNormalMap(
            const aten::MaterialParameter* param,
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            float u, float v,
            const aten::vec3& wi,
            aten::sampler* sampler);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
