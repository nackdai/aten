#pragma once

#include <memory>

#include "material/material.h"

namespace aten
{
    class Values;

    class MaterialFactory {
    private:
        MaterialFactory() = delete;
        ~MaterialFactory() = delete;

        MaterialFactory(const MaterialFactory& rhs) = delete;
        const MaterialFactory& operator=(const MaterialFactory& rhs) = delete;

    public:
        static std::shared_ptr<material> createMaterial(
            MaterialType type,
            Values& value);

        static std::shared_ptr<material> createMaterialWithDefaultValue(MaterialType type);

        static std::shared_ptr<material> createMaterialWithMaterialParameter(
            const MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);
    };
}
