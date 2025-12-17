#pragma once

#include <type_traits>
#include <utility>

#include "material/material.h"
#include "material/specular.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class refraction : public material {
        friend class material;

    private:
        refraction(
            const aten::MaterialParameter& param,
            aten::texture* normalMap = nullptr)
            : material(param, aten::MaterialAttributeRefraction)
        {
            setTextures(nullptr, normalMap, nullptr);
        }

        refraction(aten::Values& val);

        virtual ~refraction() {}

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
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler);

        virtual bool edit(aten::IParamEditor* editor) override final;

        /**
        * @brief Compute probability to sample specified output vector.
        * @return Always returns 1.0.
        */
        static inline AT_DEVICE_API float ComputePDF()
        {
            return 1.0F;
        }

        static AT_DEVICE_API void SampleRefraction(
            AT_NAME::MaterialSampling& result,
            aten::sampler* sampler,
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi);

        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float ni, const float nt,
            const aten::vec3& wo,
            const aten::vec3& n,
            const float fresnle_transmittance);
    };
}
