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
            float roughness = float(0.5),
            float ior = float(1),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::GGX, aten::MaterialAttributeMicrofacet, albedo, ior)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
            m_param.standard.roughness = aten::clamp<float>(roughness, 0, 1);
        }

        MicrofacetGGX(aten::Values& val);

        virtual ~MicrofacetGGX() {}

    public:
        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] param BRDF parameters.
         * @param[in] n Macrosurface normal.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[in] u U value of texture coordinate to sample roughness map.
         * @param[in] v V value of texture coordinate to sample roughness map.
         * @return Probability to sample output vector.
         */
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
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            float u, float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        /**
         * @brief Compute GGX microfacet distribution.
         * @param[in] m Microsurface normal.
         * @param[in] n Macrosurface normal.
         * @param[in] roughness Roughness parameter.
         * @param GGX microfacet distribution.
         */
        static AT_DEVICE_API float ComputeDistribution(
            const aten::vec3& m,
            const aten::vec3& n,
            float roughness);

        /**
         * @brief Compute G2 shadowing masking function.
         * @note Compute height correlated Smith shadowing-masking.
         * @param[in] roughness Roughness parameter.
         * @param[in] view Vector to eye.
         * @param[in] light Vector to light.
         * @param[in] n Macrosurface normal.
         * @return G2 shadowing masking value.
         */
        static AT_DEVICE_API float ComputeG2Smith(
            float roughness,
            const aten::vec3& view,
            const aten::vec3& light,
            const aten::vec3& n);

        /**
         * @brief Compute lambda function for shadowing-masking function.
         * @param[in] roughness Roughness parameter.
         * @param[in] w Target vector.
         * @param[in] m Microsurface normal.
         * @param Lambda value for shadowing-masking function.
         */
        static AT_DEVICE_API float Lambda(
            float roughness,
            const aten::vec3& w,
            const aten::vec3& n);

        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] roughness Roughness parameter.
         * @param[in] n Macrosurface normal.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @return Probability to sample output vector.
         */
        static AT_DEVICE_API float ComputeProbabilityToSampleOutputVector(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);

        /**
         * @brief Sample direction for reflection.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Macrosurface normal.
         * @param[in, out] sampler Sampler to sample
         * @return Reflect vector.
         */
        static AT_DEVICE_API aten::vec3 SampleDirection(
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n,
            aten::sampler* sampler);

        /**
         * @brief Sample microsurface normal.
         * @param[in] roughness Roughness parameter.
         * @param[in] n Macrosurface normal.
         * @param[in] r1 Rondam value by uniforma sampleing.
         * @param[in] r2 Rondam value by uniforma sampleing.
         * @return Microsurface normal.
         */
        static AT_DEVICE_API aten::vec3 SampleMicrosurfaceNormal(
            const float roughness,
            const aten::vec3& n,
            float r1, float r2);

        /**
         * @brief Compute BRDF.
         * @param[in] albedo Albedo color.
         * @param[in] roughness Roughness parameter.
         * @param[in] ior Refraction index.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[in] n Macrosurface normal.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);
    };
}
