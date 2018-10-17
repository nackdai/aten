#pragma once

#include "material/material.h"
#include "material/lambert.h"

namespace AT_NAME
{
    class emissive : public material {
        friend class MaterialFactory;

    private:
        emissive()
            : material(aten::MaterialType::Emissive, MaterialAttributeEmissive)
        {}

        emissive(const aten::vec3& e)
            : material(aten::MaterialType::Emissive, MaterialAttributeEmissive, e)
        {}

        emissive(aten::Values& val)
            : material(aten::MaterialType::Emissive, MaterialAttributeEmissive, val)
        {}

        virtual ~emissive() {}

    public:
        virtual AT_DEVICE_MTRL_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final
        {
            return emissive::pdf(&m_param, normal, wi, wo, u, v);
        }

        virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler) const override final
        {
            return std::move(emissive::sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
        }

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final
        {
            return std::move(emissive::bsdf(&m_param, normal, wi, wo, u, v));
        }

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false) const override final
        {
            MaterialSampling ret;

            sample(
                &ret,
                &m_param,
                normal,
                ray.dir,
                orgnormal,
                sampler,
                u, v,
                isLightPath);

            return std::move(ret);
        }

        static AT_DEVICE_MTRL_API real pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v)
        {
            auto ret = lambert::pdf(normal, wo);
            return ret;
        }

        static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler)
        {
            return std::move(lambert::sampleDirection(normal, sampler));
        }

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v)
        {
            auto ret = lambert::bsdf(param, u, v);
            return std::move(ret);
        }

        static AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& externalAlbedo)
        {
            auto ret = lambert::bsdf(param, externalAlbedo);
            return std::move(ret);
        }

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false)
        {
            result->dir = sampleDirection(param, normal, wi, u, v, sampler);
            result->pdf = pdf(param, normal, wi, result->dir, u, v);
            result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
        }

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false)
        {
            result->dir = sampleDirection(param, normal, wi, u, v, sampler);
            result->pdf = pdf(param, normal, wi, result->dir, u, v);
            result->bsdf = bsdf(param, externalAlbedo);
        }

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
            return real(1);
        }
    };
}
