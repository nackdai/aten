#pragma once

#include "material/material.h"

namespace AT_NAME
{
    class ToonSpecular {
        friend class material;
        friend class Toon;

    private:
        ToonSpecular() = default;
        ~ToonSpecular() = default;

    public:
        static AT_DEVICE_API float ComputePDF(
            const aten::MaterialParameter& param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API aten::vec3 ComputeBRDF(
            const aten::MaterialParameter& param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

    private:
        static AT_DEVICE_API aten::vec3 ComputeHalfVector(
            const aten::MaterialParameter& param,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo);
    };
}
