#pragma once

#include <array>
#include "material/material.h"

// NOTE
// https://www.ccs-inc.co.jp/guide/column/light_color_part2/vol04.html
// https://www.3mcompany.jp/3M/ja_JP/road-safety-jp/resources/road-transportation-safety-center-blog/full-story/~/road-signs-retroreflectivity/?storyid=70a0ffaa-7ee2-4636-ae2b-094b7c8359a6
// https://en.wikipedia.org/wiki/Retroreflector

// https://hal.inria.fr/hal-01083366/document
// https://dl.acm.org/doi/pdf/10.1145/3095140.3095176
// https://www.researchgate.net/publication/323012340_A_retroreflective_BRDF_model_based_on_prismatic_sheeting_and_microfacet_theory
// https://texeltalk.blogspot.com/2021/01/a-retro-reflective-shader-for-unity.html

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class Retroreflective : public material {
        friend class material;

    private:
        Retroreflective(
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Retroreflective, aten::MaterialAttributeMicrofacet)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

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
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false);

        static AT_DEVICE_MTRL_API void sample(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            const aten::vec3& externalAlbedo,
            bool isLightPath = false);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        static AT_DEVICE_MTRL_API real getEffectiveRetroreflectiveArea(
            const aten::vec3& into_prismatic_sheet_dir,
            const aten::vec3& normal);
    };
}
