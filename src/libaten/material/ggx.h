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
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(param, aten::MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
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
        static AT_DEVICE_API float ComputePDF(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);

        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] roughness Roughness parameter.
         * @param[in] n Macrosurface normal.
         * @param[in] m Microsurface vector(i.e. half vector).
         * @param[in] wo Output vector.
         * @return Probability to sample output vector.
         */
        static AT_DEVICE_API float ComputePDFWithHalfVector(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& m,
            const aten::vec3& wo);

        /**
         * @brief Sample direction for reflection.
         * @param[in] r1 Rondam value by uniforma sampleing.
         * @param[in] r2 Rondam value by uniforma sampleing.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Macrosurface normal.
         * @return Reflect vector.
         */
        static AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n);

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
         * @param[in] roughness Roughness parameter.
         * @param[in] ior Refraction index.
         * @param[in] n Macrosurface normal.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);

        /**
         * @brief Compute BRDF with specified half vector.
         * @param[in] roughness Roughness parameter.
         * @param[in] ior Refraction index.
         * @param[in] N Macrosurface normal.
         * @param[in] V Vector to view.
         * @param[in] L Vector to light.
         * @param[in] H Half vector.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDFWithHalfVector(
            const float roughness,
            const float ior,
            const aten::vec3& N,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& H);
    };
}
