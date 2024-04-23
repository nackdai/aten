#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class lambert : public material {
        friend class material;

    private:
        lambert(
            const aten::vec3& albedo = aten::vec3(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Lambert, aten::MaterialAttributeLambert, albedo, 0)
        {
            setTextures(albedoMap, normalMap, nullptr);
        }

        lambert(aten::Values& val);

        virtual ~lambert() {}

    public:
        static AT_DEVICE_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wo)
        {
            return ComputePDF(normal, wo);
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::vec3& normal,
            real r1, real r2)
        {
            return SampleDirection(normal, r1, r2);
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::vec3& normal,
            aten::sampler* sampler)
        {
            const auto r1 = sampler->nextSample();
            const auto r2 = sampler->nextSample();

            return SampleDirection(normal, r1, r2);
        }

        static AT_DEVICE_API aten::vec3 bsdf(const aten::MaterialParameter* param)
        {
            return ComputeBRDF();
        }

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler)
        {
            MaterialSampling ret;

            result->dir = sampleDirection(normal, sampler);
            result->pdf = pdf(normal, result->dir);
            result->bsdf = bsdf(param);
        }

        virtual bool edit(aten::IMaterialParamEditor* editor) override final
        {
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

            return AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);
        }

        /**
         * @brief Compute probability to sample specified output vector.
         * @param[in] n Surface normal.
         * @param[in] wo Output vector.
         * @return Probability to sample output vector.
         */
        static inline AT_DEVICE_API float ComputePDF(
            const aten::vec3& n,
            const aten::vec3& wo)
        {
            const auto c = aten::abs(dot(n, wo));
            const auto ret = c / AT_MATH_PI;
            return ret;
        }

        /**
         * @brief Sample direction for reflection.
         * @param[in] n Macrosurface normal.
         * @param[in] r1 Rondam value by uniform sampleing.
         * @param[in] r2 Rondam value by uniform sampleing.
         * @return Reflect vector.
         */
        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const aten::vec3& n,
            float r1, float r2)
        {
            // Importance sampling with cosine factor.
            const auto theta = aten::acos(aten::sqrt(1.0F - r1));
            const auto phi = AT_MATH_PI_2 * r2;

            const auto costheta = aten::cos(theta);
            const auto sintheta = aten::sqrt(1 - costheta * costheta);

            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sqrt(1 - cosphi * cosphi);

            auto t = aten::getOrthoVector(n);
            auto b = cross(n, t);

            auto dir = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
            dir = normalize(dir);

            AT_ASSERT(dot(n, dir) >= 0);

            return dir;
        }

        /**
         * @brief Compute BRDF.
         * @return BRDF.
         */
        static AT_DEVICE_API aten::vec3 ComputeBRDF()
        {
            return aten::vec3(1.0F) / AT_MATH_PI;
        }
    };
}
