#pragma once

#include "material/material.h"

// NOTE
// çƒãAîΩéÀ
// https://www.ccs-inc.co.jp/guide/column/light_color_part2/vol04.html
// https://www.3mcompany.jp/3M/ja_JP/road-safety-jp/resources/road-transportation-safety-center-blog/full-story/~/road-signs-retroreflectivity/?storyid=70a0ffaa-7ee2-4636-ae2b-094b7c8359a6
// https://en.wikipedia.org/wiki/Retroreflector

// https://hal.inria.fr/hal-01083366/document
// https://dl.acm.org/doi/pdf/10.1145/3095140.3095176
// https://www.researchgate.net/publication/323012340_A_retroreflective_BRDF_model_based_on_prismatic_sheeting_and_microfacet_theory

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class Retroreflective : public material {
        friend class MaterialFactory;

    private:
        Retroreflective(
            const aten::vec3& albedo = aten::vec3(0.5),
            real ior = real(0),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Specular, MaterialAttributeSpecular, albedo, ior, albedoMap, normalMap)
        {}

        Retroreflective(aten::Values& val);

        virtual ~Retroreflective() = default;

    public:
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

        static AT_DEVICE_MTRL_API void sample(
            MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            const aten::vec3& externalAlbedo,
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
            aten::sampler* sampler) const override final;

        virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual AT_DEVICE_MTRL_API MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false) const override final;

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

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
