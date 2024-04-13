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
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(1),
            bool isIdealRefraction = false,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Refraction, aten::MaterialAttributeRefraction, albedo, ior)
        {
            setTextures(nullptr, normalMap, nullptr);
            m_param.isIdealRefraction = isIdealRefraction;
        }

        refraction(aten::Values& val);

        virtual ~refraction() {}

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
            aten::sampler* sampler);

        /**
         * @brief Compute refract vector.
         * @param[in] ni Index of refraction of the media on the incident side.
         * @param[in] nt Index of refraction of the media on the transmitted side.
         * @param[in] wi Incident vector.
         * @param[in] n Normal vector on surface.
         * @return Refract vector.
         */
        static AT_DEVICE_API aten::vec3 ComputeRefractVector(
            const float ni, const float nt,
            const aten::vec3& wi,
            const aten::vec3& n)
        {
            // NOTE:
            // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
            const auto w = -wi;

            const auto costheta = aten::abs(dot(w, n));
            const auto sintheta_2 = 1.0F - costheta * costheta;

            const auto ni_nt = ni / nt;
            const auto ni_nt_2 = ni_nt * ni_nt;

            auto wo = (ni_nt * costheta - aten::sqrt(1.0F - ni_nt_2 * sintheta_2)) * n - ni_nt * w;
            wo = normalize(wo);

            return wo;
        }

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        /**
        * @brief Compute probability to sample specified output vector.
        * @return Always returns 1.0.
        */
        static inline AT_DEVICE_API float ComputeProbabilityToSampleOutputVector()
        {
            return 1.0F;
        }

        static AT_DEVICE_API void SampleRefraction(
            AT_NAME::MaterialSampling& result,
            aten::sampler* sampler,
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi);
    };
}
