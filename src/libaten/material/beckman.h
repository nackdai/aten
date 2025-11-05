#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class MicrofacetBeckman : public material {
        friend class material;
        friend class Retroreflective;

    private:
        MicrofacetBeckman(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(param, aten::MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        MicrofacetBeckman(aten::Values& val);

        virtual ~MicrofacetBeckman() {}

    public:
        static AT_DEVICE_API float pdf(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v,
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            float u, float v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const float roughness,
            const aten::vec3& in,
            const aten::vec3& normal,
            aten::sampler* sampler);

        /**
         * @brief Compute Beckman microfacet distribution.
         * @param[in] m Microsurface normal.
         * @param[in] n Macrosurface normal.
         * @param[in] roughness Roughness parameter.
         * @param Beckman microfacet distribution.
         */
        static AT_DEVICE_API float ComputeDistribution(
            const aten::vec3& m,
            const aten::vec3& n,
            float roughness);

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
         * @brief Sample direction for reflection.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Macrosurface normal.
         * @param[in] r1 Rondam value by uniforma sampleing.
         * @param[in] r2 Rondam value by uniforma sampleing.
         * @return Reflect vector.
         */
        static AT_DEVICE_API aten::vec3 SampleDirection(
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n,
            float r1, float r2);

        /**
         * @brief Compute Beckman G1 shadowing masking function.
         * @param[in] roughness Roughness parameter.
         * @param[in] v Target vector to be shadowed and masked.
         * @param[in] n Macrosurface normal.
         * @return G1 shadowing masking value.
         */
        static AT_DEVICE_API float ComputeG1(
            float roughness,
            const aten::vec3& v,
            const aten::vec3& n);

        /**
         * @brief Compute BRDF.
         * @param[in] roughness Roughness parameter.
         * @param[in] ior Refraction index.
         * @param[in] n Macrosurface normal.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[out] fresnel To obtain the computed fresnel in this API.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float* fresnel = nullptr);
    };
}
