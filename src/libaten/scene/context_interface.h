#pragma once

#include "material/material.h"
#include "light/light_parameter.h"
#include "geometry/geomparam.h"
#include "math/mat4.h"

namespace aten
{
    class context_interface {
    protected:
        context_interface() noexcept = default;
        ~context_interface() noexcept = default;

        context_interface(const context_interface&) = delete;
        context_interface(context_interface&&) = delete;

        context_interface& operator=(const context_interface&) = delete;
        context_interface& operator=(context_interface&&) = delete;

    public:
        template <typename T>
        const T GetPosition(uint32_t idx) const noexcept{ return T(); }

        template <typename T>
        const T GetNormal(uint32_t idx) const noexcept { return T(); }

        const aten::ObjectParameter& GetObject(uint32_t idx) const noexcept { return aten::ObjectParameter(); }

        const aten::MaterialParameter& GetMaterial(uint32_t idx) const noexcept { return aten::MaterialParameter(); }

        const aten::TriangleParameter& GetTriangle(uint32_t idx) const noexcept { return aten::TriangleParameter(); }

        const aten::LightParameter& GetLight(uint32_t idx) const noexcept { return aten::LightParameter(); }

        const aten::mat4& GetMatrix(uint32_t idx) const noexcept { return aten::mat4(); }
    };
}
