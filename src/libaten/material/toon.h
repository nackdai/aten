#pragma once

#include "light/light.h"
#include "material/material.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME
{
    class context;

    class Toon : public material {
        friend class material;

    protected:
        Toon(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
        {
            m_param = param;
            m_param.attrib = param.toon.toon_type == aten::MaterialType::Diffuse
                ? aten::MaterialAttributeDiffuse
                : aten::MaterialAttributeMicrofacet;
            setTextures(albedoMap, normalMap, nullptr);
        }
        ~Toon() = default;

    public:
        template <class SCENE = void>
        static inline AT_DEVICE_API aten::vec3 bsdf(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            aten::sampler& sampler,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v,
            SCENE* scene = nullptr);

        bool edit(aten::IMaterialParamEditor* editor) override;

    protected:
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            const aten::LightSampleResult* light_sample,
            aten::sampler& sampler,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v);

        static AT_DEVICE_API aten::vec3 ComputeRimLight(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi);
    };

    class StylizedBrdf : public Toon {
        friend class material;
        friend class Toon;

    private:
        StylizedBrdf(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : Toon(param, albedoMap, normalMap)
        {}
        ~StylizedBrdf() = default;

    public:
        bool edit(aten::IMaterialParamEditor* editor) override;

    protected:
        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            const aten::LightSampleResult* light_sample,
            aten::sampler& sampler,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v);
    };
}
