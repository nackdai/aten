#pragma once

#include "material/material.h"
#include "material/lambert.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class emissive : public material {
        friend class material;

    private:
        emissive()
            : material(aten::MaterialType::Emissive, aten::MaterialAttributeEmissive)
        {}

        emissive(const aten::vec3& e)
            : material(aten::MaterialType::Emissive, aten::MaterialAttributeEmissive, e)
        {}

        emissive(aten::Values& val);

        virtual ~emissive() {}

    public:

        static AT_DEVICE_API real pdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v)
        {
            auto ret = lambert::pdf(normal, wo);
            return ret;
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler)
        {
            return lambert::sampleDirection(normal, sampler);
        }

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v)
        {
            auto ret = lambert::bsdf(param, u, v);
            return ret;
        }

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& externalAlbedo)
        {
            auto ret = lambert::bsdf(param, externalAlbedo);
            return ret;
        }

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
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

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
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

        static AT_DEVICE_API real computeFresnel(
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
