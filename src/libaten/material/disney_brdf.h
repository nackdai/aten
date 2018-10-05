#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace AT_NAME
{
    class DisneyBRDF : public material {
    public:
        DisneyBRDF(
            aten::vec3 baseColor = aten::vec3(0.5),
            real subsurface = real(0.5),
            real metallic = real(0.5),
            real specular = real(0.5),
            real specularTint = real(0.5),
            real roughness = real(0.5),
            real anisotropic = real(0.5),
            real sheen = real(0.5),
            real sheenTint = real(0.5),
            real clearcoat = real(0.5),
            real clearcoatGloss = real(0.5),
            real ior = real(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Disney, MaterialAttributeMicrofacet, baseColor, 1, albedoMap, normalMap)
        {
            m_param.baseColor = baseColor;
            m_param.subsurface = aten::clamp<real>(subsurface, 0, 1);
            m_param.metallic = aten::clamp<real>(metallic, 0, 1);
            m_param.specular = aten::clamp<real>(specular, 0, 1);
            m_param.specularTint = aten::clamp<real>(specularTint, 0, 1);
            m_param.roughness = aten::clamp<real>(roughness, 0, 1);
            m_param.anisotropic = aten::clamp<real>(anisotropic, 0, 1);
            m_param.sheen = aten::clamp<real>(sheen, 0, 1);
            m_param.sheenTint = aten::clamp<real>(sheenTint, 0, 1);
            m_param.clearcoat = aten::clamp<real>(clearcoat, 0, 1);
            m_param.clearcoatGloss = aten::clamp<real>(clearcoatGloss, 0, 1);

            m_param.ior = ior;

            m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
        }

        DisneyBRDF(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Disney, MaterialAttributeMicrofacet, param.baseColor, 1, albedoMap, normalMap)
        {
            m_param.baseColor = param.baseColor;
            m_param.subsurface = aten::clamp<real>(param.subsurface, 0, 1);
            m_param.metallic = aten::clamp<real>(param.metallic, 0, 1);
            m_param.specular = aten::clamp<real>(param.specular, 0, 1);
            m_param.specularTint = aten::clamp<real>(param.specularTint, 0, 1);
            m_param.roughness = aten::clamp<real>(param.roughness, 0, 1);
            m_param.anisotropic = aten::clamp<real>(param.anisotropic, 0, 1);
            m_param.sheen = aten::clamp<real>(param.sheen, 0, 1);
            m_param.sheenTint = aten::clamp<real>(param.sheenTint, 0, 1);
            m_param.clearcoat = aten::clamp<real>(param.clearcoat, 0, 1);
            m_param.clearcoatGloss = aten::clamp<real>(param.clearcoatGloss, 0, 1);

            m_param.ior = param.ior;

            m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
        }

        DisneyBRDF(aten::Values& val)
            : material(aten::MaterialType::Disney, MaterialAttributeMicrofacet, val)
        {
            // TODO
            // Clamp parameters.
            m_param.subsurface = val.get("subsurface", m_param.subsurface);
            m_param.metallic = val.get("metallic", m_param.metallic);
            m_param.specular = val.get("specular", m_param.specular);
            m_param.specularTint = val.get("specularTint", m_param.specularTint);
            m_param.roughness = val.get("roughness", m_param.roughness);
            m_param.anisotropic = val.get("anisotropic", m_param.anisotropic);
            m_param.sheen = val.get("sheen", m_param.sheen);
            m_param.sheenTint = val.get("sheenTint", m_param.sheenTint);
            m_param.clearcoat = val.get("clearcoat", m_param.clearcoat);
            m_param.clearcoatGloss = val.get("clearcoatGloss", m_param.clearcoatGloss);
            m_param.roughnessMap = val.get("roughnessmap", m_param.roughnessMap);

            m_param.ior = val.get("ior", m_param.ior);
        }

        virtual ~DisneyBRDF() {}

    public:
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

        static AT_DEVICE_MTRL_API real pdf(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath);

        virtual AT_DEVICE_MTRL_API real computeFresnel(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor = 1) const
        {
            return computeFresnel(&m_param, normal, wi, wo, outsideIor);
        }

        static AT_DEVICE_MTRL_API real computeFresnel(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
