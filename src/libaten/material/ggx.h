#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class MicrofacetGGX : public material {
        friend class MicrofacetRefraction;
        friend class material;

    private:
        MicrofacetGGX(
            const aten::vec3& albedo = aten::vec3(0.5),
            real roughness = real(0.5),
            real ior = real(1),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, albedo, ior)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
            m_param.standard.roughness = aten::clamp<real>(roughness, 0, 1);
        }

        MicrofacetGGX(aten::Values& val);

        virtual ~MicrofacetGGX() {}

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

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        static AT_DEVICE_MTRL_API real sampleGGX_D(
            const aten::vec3& wh,    // half
            const aten::vec3& n,    // normal
            real roughness);

        static AT_DEVICE_MTRL_API real computeGGXSmithG1(
            real roughness,
            const aten::vec3& v,
            const aten::vec3& n);

    private:
        static AT_DEVICE_MTRL_API real pdf(
            const real roughness,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo);

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const real roughness,
            const aten::vec3& in,
            const aten::vec3& normal,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 sampleNormal(
            const real roughness,
            const aten::vec3& normal,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& albedo,
            const real roughness,
            const real ior,
            real& fresnel,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);
    };
}
