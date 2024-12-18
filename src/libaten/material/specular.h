#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class specular : public material {
        friend class material;

    private:
        specular(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(param, aten::MaterialAttributeSpecular)
        {
            setTextures(albedoMap, normalMap, nullptr);
        }

        specular(aten::Values& val);

        virtual ~specular() {}

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
            aten::sampler* sampler,
            float u, float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        /**
         * @brief Compute probability to sample specified output vector.
         * @return Always returns 1.0.
         */
        static inline AT_DEVICE_API float ComputePDF()
        {
            return 1.0F;
        }

        /**
         * @brief Sample direction for reflection.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Surface normal.
         * @return Reflect vector.
         */
        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const aten::vec3& wi,
            const aten::vec3& n)
        {
            const auto wo = material::ComputeReflectVector(wi, n);
            return wo;
        }

        /**
         * @brief Compute BRDF.
         * @param[in] wo Output vector.
         * @param[in] n Surface normal.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const aten::vec3& wo,
            const aten::vec3& n)
        {
            // NOTE:
            // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularReflection

            // For canceling cosine factor.
            const auto c = aten::dot(n, wo);
            const auto bsdf = c == 0.0F ? 0.0F : 1.0F / c;
            return aten::vec3(bsdf);
        }
    };
}
