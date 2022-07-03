#pragma once

#include <vector>
#include "material/material.h"

namespace AT_NAME
{
    class LayeredBSDF : public material {
    public:
        LayeredBSDF()
            : material(aten::MaterialType::Layer, MaterialAttributeMicrofacet)
        {}
        virtual ~LayeredBSDF() {}

    public:
        bool add(const std::shared_ptr<material>& mtrl);

        virtual aten::vec4 sampleAlbedoMap(real u, real v) const override final;

        virtual bool isGlossy() const override final;

        virtual real applyNormalMap(
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            real u, real v,
            const aten::vec3& wi,
            aten::sampler* sampler) const override final;

        virtual real computeFresnel(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor = 1) const override final;

        virtual real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler,
            real pre_sampled_r) const override final;

        virtual aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r) const override final;

        virtual MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            bool isLightPath = false) const override final;

    private:
        std::vector<std::shared_ptr<material>> m_layer;
    };
}
