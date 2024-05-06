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
            aten::vec3 albedo = aten::vec3(0.5F),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Retroreflective, aten::MaterialAttributeMicrofacet, albedo, 0)
        {
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        Retroreflective(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr,
            aten::texture* roughnessMap = nullptr)
            : material(aten::MaterialType::Retroreflective, aten::MaterialAttributeMicrofacet)
        {
            m_param.baseColor = param.baseColor;
            m_param.standard = param.standard;
            setTextures(albedoMap, normalMap, roughnessMap);
        }

        Retroreflective(aten::Values& val);

        virtual ~Retroreflective() = default;

    public:
        static AT_DEVICE_API real pdf(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler);

        static AT_DEVICE_API aten::vec3 bsdf(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_API void sample(
            AT_NAME::MaterialSampling& result,
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            aten::sampler* sampler,
            real u, real v);

        virtual bool edit(aten::IMaterialParamEditor* editor) override final;

        enum Component {
            SurfaceReflection,
            RetroReflection,
            Diffuse,
            Num = Diffuse + 1,
        };

        static AT_DEVICE_API void ComputeWeights(
            std::array<float, Component::Num>& weights,
            const float ni, const float nt,
            const aten::vec3& wi,
            const aten::vec3& n);

        static AT_DEVICE_API void GetCDF(
            const std::array<float, Component::Num>& weights,
            std::array<float, Component::Num>& cdf);

        static AT_DEVICE_API float ComputePDF(
            const aten::MaterialParameter& param,
            const aten::vec3& wi,
            const aten::vec3& wo,
            const aten::vec3& n);

        static AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2, const float r3,
            const aten::MaterialParameter& param,
            const aten::vec3& wi,
            const aten::vec3& n);

        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const aten::MaterialParameter& param,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo);
    };
}
