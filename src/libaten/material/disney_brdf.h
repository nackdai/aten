#pragma once

#include "material/material.h"
#include "image/texture.h"

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
            : material(aten::MaterialParameter(), aten::MaterialAttributeMicrofacet)
        {
            param_.type = aten::MaterialType::Disney;
            param_.baseColor = baseColor;
            param_.standard.subsurface = aten::saturate(subsurface);
            param_.standard.metallic = aten::saturate(metallic);
            param_.standard.specular = aten::saturate(specular);
            param_.standard.specularTint = aten::saturate(specularTint);
            param_.standard.roughness = aten::saturate(roughness);
            param_.standard.anisotropic = aten::saturate(anisotropic);
            param_.standard.sheen = aten::saturate(sheen);
            param_.standard.sheenTint = aten::saturate(sheenTint);
            param_.standard.clearcoat = aten::saturate(clearcoat);
            param_.standard.clearcoatGloss = aten::saturate(clearcoatGloss);

            param_.standard.ior = ior;

            SetTextures(albedoMap, normalMap, roughnessMap);
        }

        DisneyBRDF(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(param, aten::MaterialAttributeMicrofacet)
        {
            param_.baseColor = param.baseColor;
            param_.standard.subsurface = aten::saturate(param.standard.subsurface);
            param_.standard.metallic = aten::saturate(param.standard.metallic);
            param_.standard.specular = aten::saturate(param.standard.specular);
            param_.standard.specularTint = aten::saturate(param.standard.specularTint);
            param_.standard.roughness = aten::saturate(param.standard.roughness);
            param_.standard.anisotropic = aten::saturate(param.standard.anisotropic);
            param_.standard.sheen = aten::saturate(param.standard.sheen);
            param_.standard.sheenTint = aten::saturate(param.standard.sheenTint);
            param_.standard.clearcoat = aten::saturate(param.standard.clearcoat);
            param_.standard.clearcoatGloss = aten::saturate(param.standard.clearcoatGloss);

            param_.standard.ior = param.standard.ior;

            SetTextures(albedoMap, normalMap, roughnessMap);
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

        static AT_DEVICE_API AT_NAME::MaterialSampling bsdf(
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
            const aten::vec3& base_color,
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
