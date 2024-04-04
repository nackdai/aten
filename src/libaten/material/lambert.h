#pragma once

#include "material/material.h"
#include "texture/texture.h"
#include "material/sample_texture.h"

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
        static AT_HOST_DEVICE_API real pdf(
            const aten::vec3& normal,
            const aten::vec3& wo)
        {
            const auto c = aten::abs(dot(normal, wo));
            const auto ret = c / AT_MATH_PI;
            return ret;
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::vec3& normal,
            real r1, real r2)
        {
            auto n = normal;
            auto t = aten::getOrthoVector(n);
            auto b = cross(n, t);

            // Importance sampling with cosine factor.
            r1 = 2 * AT_MATH_PI * r1;
            const real r2s = sqrt(r2);

            const real x = aten::cos(r1) * r2s;
            const real y = aten::sin(r1) * r2s;
            const real z = aten::sqrt(real(1) - r2);

            aten::vec3 dir = normalize((t * x + b * y + n * z));
            AT_ASSERT(dot(normal, dir) >= 0);

            return dir;
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::vec3& normal,
            aten::sampler* sampler)
        {
            const auto r1 = sampler->nextSample();
            const auto r2 = sampler->nextSample();

            return sampleDirection(normal, r1, r2);
        }

        static AT_DEVICE_API aten::vec3 bsdf(const aten::MaterialParameter* param)
        {
            aten::vec3 ret = aten::vec3(1.0f) / AT_MATH_PI;
            return ret;
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
    };
}
