#pragma once

#include "material/material.h"

namespace AT_NAME
{
    class context;

    class Toon : public material {
        friend class material;

    private:
        Toon(
            const aten::vec3& albedo = aten::vec3(0.5),
            aten::texture* albedoMap = nullptr,
            aten::texture* normalMap = nullptr)
            : material(aten::MaterialType::Toon, aten::MaterialAttributeDiffuse, albedo, 0)
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

        bool edit(aten::IMaterialParamEditor* editor) override final;
    };
}
