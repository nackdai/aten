#pragma once

#include "material/material.h"

namespace AT_NAME
{
    class context;

    class Toon : public material {
        friend class material;

    private:
        Toon(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(param, aten::MaterialAttributeDiffuse)
        {
            setTextures(albedoMap, normalMap, nullptr);
        }
        ~Toon() = default;

    public:
        static AT_DEVICE_API aten::vec3 bsdf(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            aten::sampler& sampler,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi,
            float u, float v,
            float* pdf = nullptr);

        static AT_DEVICE_API aten::vec3 PostProcess(
            const AT_NAME::context& ctxt,
            const aten::MaterialParameter& param,
            const aten::vec3& hit_pos,
            const aten::vec3& normal,
            const aten::vec3& wi);

        bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
