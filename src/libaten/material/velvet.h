#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class MicrofacetVelvet : public material {
        friend class material;

    private:
        MicrofacetVelvet(
            const aten::vec3& albedo = aten::vec3(0.5),
            float roughness = float(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Velvet, aten::MaterialAttributeMicrofacet, albedo, 0)
        {
            setTextures(albedoMap, normalMap, nullptr);
            m_param.standard.roughness = aten::clamp<float>(roughness, 0, 1);
        }

        MicrofacetVelvet(aten::Values& val);

        virtual ~MicrofacetVelvet() {}

    public:
        static AT_DEVICE_API float pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& n,
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
            aten::sampler* sampler,
            float u, float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        /**
         * @brief Compute velvet microfacet distribution.
         * @param[in] m Microsurface normal.
         * @param[in] n Macrosurface normal.
         * @param[in] roughness Roughness parameter.
         * @param Velvet microfacet distribution.
         */
        static AT_DEVICE_API float ComputeDistribution(
            const aten::vec3& m,
            const aten::vec3& n,
            const float roughness);

        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] n Surface normal.
         * @param[in] wo Output vector.
         * @return Probability to sample output vector.
         */
        static AT_DEVICE_API float ComputePDF(
            const aten::vec3& n,
            const aten::vec3& wo);

        /**
         * @brief Sample direction for reflection.
         * @param[in] n Macrosurface normal.
         * @param[in] r1 Rondam value by uniforma sampleing.
         * @param[in] r2 Rondam value by uniforma sampleing.
         * @return Reflect vector.
         */
        static AT_DEVICE_API aten::vec3 SampleDirection(
            const aten::vec3& n,
            float r1, float r2);

        /**
         * @brief Compute lambda function for shadowing-masking function.
         * @param[in] roughness Roughness parameter.
         * @param[in] w Target vector.
         * @param[in] m Microsurface normal.
         * @param Lambda value for shadowing-masking function.
         */
        static AT_DEVICE_API float ComputeVelvetLambda(
            const float roughness,
            const aten::vec3& w,
            const aten::vec3& m);

        /**
         * @brief Compute velvet L factor to compute lambda function for shadowing-masking function.
         * @note L factor is: L(x) = a / (1 + b * x^c) + d * x + e
         * @param[in] x Argument for L factor.
         * @param[in] roughness Surface roughness.
         * @return Velvel L factor to compute lambda function for shadowing-masking function.
         */
        static AT_DEVICE_API float ComputeVelvetLForLambda(const float x, const float roughness);

        /**
         * @brief Interpolate velvet parameter.
         * @param[in] idx Index of paramter table.
         * @param[in] interp_factor Factor to interpolate.
         * @return Interpolated velvet parameter.
         */
        static AT_DEVICE_API float InterpolateVelvetParam(const int32_t idx, float interp_factor);

        /**
         * @brief Compute G2 shadowing masking function.
         * @param[in] roughness Roughness parameter.
         * @param[in] view Vector to eye.
         * @param[in] light Vector to light.
         * @param[in] n Macrosurface normal.
         * @return G2 shadowing masking value.
         */
        static AT_DEVICE_API float ComputeShadowingMaskingFunction(
            const float roughness,
            const aten::vec3& view,
            const aten::vec3& light,
            const aten::vec3& n);

        /**
         * @brief Compute BRDF.
         * @param[in] albedo Albedo color.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[in] n Macrosurface normal.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);
    };
}
