#pragma once

#include "material/material.h"

namespace aten
{
    class MaterialFactory {
    private:
        MaterialFactory() = delete;
        ~MaterialFactory() = delete;

        MaterialFactory(const MaterialFactory& rhs) = delete;
        const MaterialFactory& operator=(const MaterialFactory& rhs) = delete;

    public:
        static material* createMaterial(
            MaterialType type,
            Values& value);

        static material* createMaterialWithDefaultValue(MaterialType type);

        static material* createMaterialWithMaterialParameter(
            MaterialType type,
            const MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);
    };
}
