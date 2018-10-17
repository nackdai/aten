#pragma once

#include "material/material.h"

namespace AT_NAME
{
    class CarPaintBRDF : public material {
        friend class MaterialFactory;

    private:
        CarPaintBRDF(
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(1),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* flakesMap = nullptr)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
        {
            m_param.roughnessMap = flakesMap ? flakesMap->id() : -1;

            m_param.carpaint.clearcoatRoughness = real(0.5);
            m_param.carpaint.flakeLayerRoughness = real(0.5);

            m_param.carpaint.flake_scale = real(100);
            m_param.carpaint.flake_size = real(0.01);
            m_param.carpaint.flake_size_variance = real(0.25);
            m_param.carpaint.flake_normal_orientation = real(0.5);
            
            m_param.carpaint.flake_reflection = real(0.5);
            m_param.carpaint.flake_transmittance = real(0.5);

            m_param.carpaint.glitterColor = albedo;
            m_param.carpaint.flakeColor = albedo;

            m_param.carpaint.flake_intensity = real(1);
        }

        CarPaintBRDF(
            const aten::MaterialParameter& param,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, param.baseColor, param.ior, nullptr, nullptr)
        {
            // TODO
            // Clamp parameters.
            m_param.carpaint.clearcoatRoughness = param.carpaint.clearcoatRoughness;
            m_param.carpaint.flakeLayerRoughness = param.carpaint.flakeLayerRoughness;
            m_param.carpaint.flake_scale =  param.carpaint.flake_scale;
            m_param.carpaint.flake_size = param.carpaint.flake_size;
            m_param.carpaint.flake_size_variance = param.carpaint.flake_size_variance;
            m_param.carpaint.flake_normal_orientation = param.carpaint.flake_normal_orientation;
            m_param.carpaint.flake_reflection = param.carpaint.flake_reflection;
            m_param.carpaint.flake_transmittance = param.carpaint.flake_transmittance;
            m_param.carpaint.glitterColor = param.carpaint.glitterColor;
            m_param.carpaint.flakeColor = param.carpaint.flakeColor;
            m_param.carpaint.flake_intensity = param.carpaint.flake_intensity;

            m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
        }

        CarPaintBRDF(aten::Values& val)
            : material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, val)
        {
            // TODO
            // Clamp parameters.
            m_param.carpaint.clearcoatRoughness = val.get("clearcoatRoughness", m_param.carpaint.clearcoatRoughness);
            m_param.carpaint.flakeLayerRoughness = val.get("flakeLayerRoughness", m_param.carpaint.flakeLayerRoughness);
            m_param.carpaint.flake_scale = val.get("flake_scale", m_param.carpaint.flake_scale);
            m_param.carpaint.flake_size = val.get("flake_size", m_param.carpaint.flake_size);
            m_param.carpaint.flake_size_variance = val.get("flake_size_variance", m_param.carpaint.flake_size_variance);
            m_param.carpaint.flake_normal_orientation = val.get("flake_normal_orientation", m_param.carpaint.flake_normal_orientation);
            m_param.carpaint.flake_reflection = val.get("flake_reflection", m_param.carpaint.flake_reflection);
            m_param.carpaint.flake_transmittance = val.get("flake_transmittance", m_param.carpaint.flake_transmittance);
            m_param.carpaint.glitterColor = val.get("glitterColor", m_param.carpaint.glitterColor);
            m_param.carpaint.flakeColor = val.get("clearcoat", m_param.carpaint.flakeColor);
            m_param.carpaint.flake_intensity = val.get("clearcoatGloss", m_param.carpaint.flake_intensity);

            auto roughnessMap = (aten::texture*)val.get("roughnessmap", nullptr);
            m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
        }

        virtual ~CarPaintBRDF() {}

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
            aten::sampler* sampler) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal, 
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false) const override final;

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
