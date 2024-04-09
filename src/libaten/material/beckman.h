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
            const aten::vec3& albedo = aten::vec3(0.5),
            real roughness = real(0.5),
            real ior = real(1),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Beckman, aten::MaterialAttributeMicrofacet, albedo, ior)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
            m_param.standard.roughness = aten::clamp<real>(roughness, 0, 1);
        }

        MicrofacetBeckman(aten::Values& val);

        virtual ~MicrofacetBeckman() {}

    public:
        static AT_DEVICE_API real pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const real roughness,
            const aten::vec3& in,
            const aten::vec3& normal,
            aten::sampler* sampler);

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
            const auto costheta = aten::abs(dot(m, n));
            if (costheta <= 0) {
                return 0;
            }

            const auto cos2 = costheta * costheta;
            const auto cos4 = cos2 * cos2;

            const auto sintheta = aten::sqrt(1 - cos2);
            const auto tantheta = sintheta / costheta;
            const auto tan2 = tantheta * tantheta;

            const auto a = roughness;
            const auto a2 = a * a;

            auto D = 1.0f / (AT_MATH_PI * a2 * cos4);
            D *= aten::exp(-tan2 / a2);

            return D;
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
            const auto wh = normalize(-wi + wo);

            const auto costheta = aten::abs(dot(wh, n));

            auto D = ComputeDistribution(wh, n, roughness);

            // For Jacobian |dwh/dwo|
            const auto denom = 4 * aten::abs(dot(wo, wh));

            const auto pdf = denom > 0 ? (D * costheta) / denom : 0;

            return pdf;
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
            const auto a2 = a * a;

            // NOTE:
            // log can't accept zero. If r1 is 1, (1 - r1) is zero.
            // To avoid it, if r1 is 1, to lessen r1 less than 1, multipley 0.99.
            const auto theta = aten::atan(aten::sqrt(-a2 * aten::log(1.0F - r1 * 0.99F)));

            const auto phi = AT_MATH_PI_2 * r2;

            const auto costheta = aten::cos(theta);
            const auto sintheta = aten::sqrt(1.0F - costheta * costheta);

            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sin(phi);

            // Ortho normal base.
            const auto t = aten::getOrthoVector(n);
            const auto b = normalize(cross(n, t));

            auto m = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
            m = normalize(m);

            return m;
        }

        /**
         * @brief Sample direction for reflection.
         * @param[in] roughness Roughness parameter.
         * @param[in] wi Incident vector.
         * @param[in] n Macrosurface normal.
         * @param[in] r1 Rondam value by uniforma sampleing.
         * @param[in] r2 Rondam value by uniforma sampleing.
         * @return Reflect vector.
         */
        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n,
            float r1, float r2)
        {
            const auto m = SampleMicrosurfaceNormal(roughness, n, r1, r2);

            // We can assume ideal reflection on each micro surface.
            // So, compute ideal reflection vector based on micro surface normal.
            const auto wo = material::ComputeReflectVector(wi, m);

            return wo;
        }

        /**
         * @brief Compute Beckman G1 shadowing masking function.
         * @note Compute height correlated Smith shadowing-masking.
         * @param[in] roughness Roughness parameter.
         * @param[in] v Target vector to be shadowed and masked.
         * @param[in] n Macrosurface normal.
         * @param[in] m Microsurface normal.
         * @return G1 shadowing masking value.
         */
        static AT_DEVICE_API real ComputeG1(
            float roughness,
            const aten::vec3& v,
            const aten::vec3& m,
            const aten::vec3& n)
        {
            const auto costheta = aten::abs(dot(v, n));
            const auto sintheta = aten::sqrt(1.0F - costheta * costheta);
            const auto tantheta = sintheta / costheta;

            const auto a = 1.0F / (roughness * tantheta);
            const auto a2 = a * a;

            if (a < 1.6F) {
                return (3.535F * a + 2.181F * a2) / (1.0F + 2.276F * a + 2.577F * a2);
            }
            else {
                return 1.0F;
            }
        }

        /**
         * @brief Compute BRDF.
         * @param[in] albedo Albedo color.
         * @param[in] roughness Roughness parameter.
         * @param[in] ior Refraction index.
         * @param[in] wi Incident vector.
         * @param[in] wo Output vector.
         * @param[in] n Macrosurface normal.
         * @param[out] fresnel To obtain the computed fresnel in this API.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float* fresnel = nullptr)
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
            const auto G2 = ComputeG1(roughness, V, H, N) * ComputeG1(roughness, L, H, N);
            const auto F = material::ComputeSchlickFresnel(ni, nt, L, H);

            if (fresnel) {
                *fresnel = F;
            }

            const auto denom = 4 * NL * NV;

            const auto bsdf = denom > AT_MATH_EPSILON ? F * G2 * D / denom : 0.0f;

            return aten::vec3(bsdf);
        }
    };
}
