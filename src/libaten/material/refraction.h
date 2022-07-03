#pragma once

#include "material/material.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class refraction : public material {
        friend class MaterialFactory;

    private:
        refraction(
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(1),
            bool isIdealRefraction = false,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Refraction, MaterialAttributeRefraction, albedo, ior, nullptr, normalMap)
        {
            m_param.isIdealRefraction = isIdealRefraction;
        }

        refraction(aten::Values& val);

        virtual ~refraction() {}

    public:
        void setIsIdealRefraction(bool f)
        {
            m_param.isIdealRefraction = f;
        }
        bool isIdealRefraction() const
        {
            return (m_param.isIdealRefraction > 0);
        }

        static AT_DEVICE_MTRL_API real pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            const aten::vec3& externalAlbedo);

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false);

        virtual AT_DEVICE_MTRL_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler,
            real pre_sampled_r) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r) const override final;

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
            real u, real v,
            bool isLightPath = false) const override final;

        struct RefractionSampling {
            bool isRefraction;
            bool isIdealRefraction;
            real probReflection;
            real probRefraction;

            RefractionSampling(bool _isRefraction, real _probReflection, real _probRefraction, bool _isIdealRefraction = false)
                : isRefraction(_isRefraction), probReflection(_probReflection), probRefraction(_probRefraction),
                isIdealRefraction(_isIdealRefraction)
            {}
        };

        static RefractionSampling check(
            const material* mtrl,
            const aten::vec3& in,
            const aten::vec3& normal,
            const aten::vec3& orienting_normal);

        virtual AT_DEVICE_MTRL_API real computeFresnel(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor = 1) const override final
        {
            return computeFresnel(&m_param, normal, wi, wo, outsideIor);
        }

        static AT_DEVICE_MTRL_API real computeFresnel(
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real outsideIor)
        {
            // TODO
            AT_ASSERT(false);
            return real(0);
        }

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
