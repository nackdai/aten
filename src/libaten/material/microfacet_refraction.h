#pragma once

#include <optional>

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    // NOTE
    // Microfacet Models for Refraction through Rough Surfaces
    // https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    // https://agraphicsguy.wordpress.com/2015/11/11/glass-material-simulated-by-microfacet-bxdf/

    class MicrofacetRefraction : public material {
        friend class material;

    private:
        MicrofacetRefraction(
            const aten::vec3& albedo = aten::vec3(0.5),
            const float roughness = 0.5F,
            const float ior = 1.0F,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Microfacet_Refraction, aten::MaterialAttributeRefraction, albedo, ior)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        MicrofacetRefraction(aten::Values& val);

    public:
        static AT_DEVICE_API float pdf(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            const float u, const float v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const float u, const float v,
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            const float u, const float v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling& result,
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            aten::sampler* sampler,
            const float u, const float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        static AT_DEVICE_API void SampleMicrofacetRefraction(
            AT_NAME::MaterialSampling& result,
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            aten::sampler* sampler);
    };
}
