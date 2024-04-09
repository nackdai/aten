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
            float roughness)
        {
            const auto a = roughness;
            const auto a2 = a * a;

            const auto costheta = aten::abs(dot(m, n));
            const auto cos2 = costheta * costheta;

            const auto denom = (a2 - 1) * cos2 + 1.0f;
            const auto denom2 = denom * denom;

            const auto D = denom > 0 ? a2 / (AT_MATH_PI * denom2) : 0;

            return D;
        }

        /**
         * @brief Compute G2 shadowing masking function.
         * @note Compute height correlated Smith shadowing-masking.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[in] m Microsurface normal.
         * @return G2 shadowing masking value.
         */
        static AT_DEVICE_API float ComputeG2Smith(
            float roughness,
            const aten::vec3& wi,
            const aten::vec3& wo,
            const aten::vec3& m)
        {
            const auto lambda_wi = Lambda(roughness, wi, m);
            const auto lambda_wo = Lambda(roughness, wo, m);
            const auto g2 = 1.0f / (1.0f + lambda_wi + lambda_wo);
            return g2;
        }

    private:
        /**
         * @brief Compute lambda function for shadowing-masking function.
         * @param[in] roughness Roughness parameter.
         * @param[in] w Target vector.
         * @param[in] m Microsurface normal.
         * @param Lambda value for shadowing-masking function.
         */
        static inline AT_DEVICE_API float Lambda(
            float roughness,
            const aten::vec3& w,
            const aten::vec3& n)
        {
            const auto alpha = roughness;

            const auto cos_theta = aten::abs(dot(w, n));
            const auto cos2 = cos_theta * cos_theta;

            const auto sin2 = 1.0f - cos2;
            const auto tan2 = sin2 / cos2;

            const auto a2 = 1.0f / (alpha * alpha * tan2);

            const auto lambda = (-1.0f + aten::sqrt(1.0f + 1.0f / a2)) / 2.0f;

            return lambda;
        }

        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] roughness Roughness parameter.
         * @param[in] n Macrosurface normal.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @return Probability to sample output vector.
         */
        static inline AT_DEVICE_API float ComputeProbabilityToSampleOutputVector(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            auto wh = normalize(-wi + wo);

            auto D = ComputeDistribution(wh, n, roughness);

            auto costheta = aten::abs(dot(wh, n));

            // For Jacobian |dwh/dwo|
            auto denom = 4 * aten::abs(dot(wo, wh));

            auto pdf = denom > 0 ? (D * costheta) / denom : 0;

            return pdf;
        }

        /**
         * @brief Sample direction for reflection.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Macrosurface normal.
         * @param[in, out] sampler Sampler to sample
         * @return Reflect vector.
         */
        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n,
            aten::sampler* sampler)
        {
            const auto r1 = sampler->nextSample();
            const auto r2 = sampler->nextSample();

            const auto m = SampleMicrosurfaceNormal(roughness, n, r1, r2);

            // We can assume ideal reflection on each micro surface.
            // So, compute ideal reflection vector based on micro surface normal.
            const auto wo = material::ComputeReflectVector(wi, m);

            return wo;
        }

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
            float r1, float r2)
        {
            const auto a = roughness;

            auto theta = aten::atan(a * aten::sqrt(r1 / (1 - r1)));
            theta = ((theta >= 0) ? theta : (theta + 2 * AT_MATH_PI));

            const auto phi = 2 * AT_MATH_PI * r2;

            const auto costheta = aten::cos(theta);
            const auto sintheta = aten::sqrt(1 - costheta * costheta);

            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sqrt(1 - cosphi * cosphi);

            // Ortho normal base.
            const auto t = aten::getOrthoVector(n);
            const auto b = normalize(cross(n, t));

            auto m = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
            m = normalize(m);

            return m;
        }

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
            const aten::vec3& wo)
        {
            aten::vec3 V = -wi;
            aten::vec3 L = wo;
            aten::vec3 N = n;

            // We can assume ideal reflection on each micro surface.
            // It means wo (= L) is computed as ideal reflection vector.
            // Then, we can compute micro surface normal as the half vector between incident and output vector.
            aten::vec3 H = normalize(L + V);

            auto NH = aten::abs(dot(N, H));
            auto VH = aten::abs(dot(V, H));
            auto NL = aten::abs(dot(N, L));
            auto NV = aten::abs(dot(N, V));

            // Assume index of refraction of the medie on the incident side is vacuum.
            const auto ni = 1.0F;
            const auto nt = ior;

            const auto D = ComputeDistribution(H, N, roughness);
            const auto G2 = ComputeG2Smith(roughness, V, L, H);
            const auto F = material::ComputeSchlickFresnel(ni, nt, L, H);

            const auto denom = 4 * NL * NV;

            const auto bsdf = denom > AT_MATH_EPSILON ? F * G2 * D / denom : 0.0f;

            return aten::vec3(bsdf);
        }
    };
}
