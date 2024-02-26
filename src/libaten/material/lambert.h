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
            auto c = dot(normal, wo);
            //AT_ASSERT(c >= 0);
            c = aten::abs(c);

            auto ret = c / AT_MATH_PI;

            return ret;
        }

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::vec3& normal,
            real r1, real r2)
        {
            // normalの方向を基準とした正規直交基底(w, u, v)を作る.
            // この基底に対する半球内で次のレイを飛ばす.
#if 1
            auto n = normal;
            auto t = aten::getOrthoVector(n);
            auto b = cross(n, t);
#else
            aten::vec3 n, t, b;

            n = normal;

            // nと平行にならないようにする.
            if (fabs(n.x) > AT_MATH_EPSILON) {
                t = normalize(cross(aten::vec3(0.0, 1.0, 0.0), n));
            }
            else {
                t = normalize(cross(aten::vec3(1.0, 0.0, 0.0), n));
            }
            b = cross(n, t);
#endif

            // コサイン項を使った重点的サンプリング.
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

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            real u, real v)
        {
            auto albedo = param->baseColor;
            albedo *= sampleTexture(
                param->albedoMap,
                u, v,
                aten::vec4(real(1)));

            aten::vec3 ret = albedo / AT_MATH_PI;
            return ret;
        }

        static AT_HOST_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter* param,
            const aten::vec3& externalAlbedo)
        {
            aten::vec3 albedo = param->baseColor;
            albedo *= externalAlbedo;

            aten::vec3 ret = albedo / AT_MATH_PI;
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
            MaterialSampling ret;

            result->dir = sampleDirection(normal, sampler);
            result->pdf = pdf(normal, result->dir);
            result->bsdf = bsdf(param, u, v);
        }

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false)
        {
            MaterialSampling ret;

            result->dir = sampleDirection(normal, sampler);
            result->pdf = pdf(normal, result->dir);
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

        virtual bool edit(aten::IMaterialParamEditor* editor) override final
        {
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
            AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

            return AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);
        }
    };
}
