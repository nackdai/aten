#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class DisneyBRDF : public material {
        friend class material;

    private:
        DisneyBRDF(
            aten::vec3 baseColor = aten::vec3(0.5),
            float subsurface = float(0.5),
            float metallic = float(0.5),
            float specular = float(0.5),
            float specularTint = float(0.5),
            float roughness = float(0.5),
            float anisotropic = float(0.5),
            float sheen = float(0.5),
            float sheenTint = float(0.5),
            float clearcoat = float(0.5),
            float clearcoatGloss = float(0.5),
            float ior = float(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Disney, aten::MaterialAttributeMicrofacet, baseColor, 1)
        {
            m_param.baseColor = baseColor;
            m_param.standard.subsurface = aten::saturate(subsurface);
            m_param.standard.metallic = aten::saturate(metallic);
            m_param.standard.specular = aten::saturate(specular);
            m_param.standard.specularTint = aten::saturate(specularTint);
            m_param.standard.roughness = aten::saturate(roughness);
            m_param.standard.anisotropic = aten::saturate(anisotropic);
            m_param.standard.sheen = aten::saturate(sheen);
            m_param.standard.sheenTint = aten::saturate(sheenTint);
            m_param.standard.clearcoat = aten::saturate(clearcoat);
            m_param.standard.clearcoatGloss = aten::saturate(clearcoatGloss);

            m_param.standard.ior = ior;

            setTextures(albedoMap, normalMap, roughnessMap);
        }

        DisneyBRDF(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Disney, aten::MaterialAttributeMicrofacet, param.baseColor, 1)
        {
            m_param.baseColor = param.baseColor;
            m_param.standard.subsurface = aten::saturate(param.standard.subsurface);
            m_param.standard.metallic = aten::saturate(param.standard.metallic);
            m_param.standard.specular = aten::saturate(param.standard.specular);
            m_param.standard.specularTint = aten::saturate(param.standard.specularTint);
            m_param.standard.roughness = aten::saturate(param.standard.roughness);
            m_param.standard.anisotropic = aten::saturate(param.standard.anisotropic);
            m_param.standard.sheen = aten::saturate(param.standard.sheen);
            m_param.standard.sheenTint = aten::saturate(param.standard.sheenTint);
            m_param.standard.clearcoat = aten::saturate(param.standard.clearcoat);
            m_param.standard.clearcoatGloss = aten::saturate(param.standard.clearcoatGloss);

            m_param.standard.ior = param.standard.ior;

            setTextures(albedoMap, normalMap, roughnessMap);
        }

        DisneyBRDF(aten::Values& val);

        virtual ~DisneyBRDF() {}

    public:
        static AT_DEVICE_API float pdf(
            const aten::MaterialParameter& mtrl,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter& mtrl,
            const aten::vec3& n,
            const aten::vec3& wi,
            float u, float v,
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter& mtrl,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling& result,
            const aten::MaterialParameter& mtrl,
            const aten::vec3& n,
            const aten::vec3& wi,
            aten::sampler* sampler,
            float u, float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        enum Component {
            Diffuse,
            Sheen,
            Specular,
            Clearcoat,
            Num = Clearcoat + 1,
        };

        static AT_DEVICE_API float ComputePDF(
            const aten::MaterialParameter& param,
            const aten::vec3& wi,
            const aten::vec3& wo,
            const aten::vec3& n);

        static AT_DEVICE_API void ComputeWeights(
            std::array<float, Component::Num>& weights,
            const float metalic,
            const float sheen,
            const float specular,
            const float clearcoat);

        static AT_DEVICE_API void GetCDF(
            const std::array<float, Component::Num>& weights,
            std::array<float, Component::Num>& cdfs);

        static AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2, const float r3,
            const aten::MaterialParameter& param,
            const aten::vec3& wi,
            const aten::vec3& n);

        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);
    };
}
