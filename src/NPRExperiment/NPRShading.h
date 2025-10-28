#pragma once

#include <vector>

#include "NPRModule.h"

class NPRShading : public NPRModule {
public:
    NPRShading() = default;
    ~NPRShading() = default;

    void InitDebugVisual(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS) override final
    {}

    void PreRender(aten::shader& shader, const aten::PinholeCamera& camera) override final;

    void DrawDebugVisual(
        const aten::context& ctxt,
        const aten::Camera& cam)
    {}

    void EditParameter() override final;

private:
    // Umbra to penumbra separation threshold.
    float shadow_level_{ 0.15F };

    // Penumbra to highlights separation threshold.
    float midtone_level_{ 0.3F };

    // Softness of tonal range transition, higher values produce softner results, lower values produce sharper transitions
    float diffuse_softness_{ 0.05F };

    // Global diffuse term tint.
    aten::vec3 diffuse_tint_{ 1, 1, 1 };

    aten::vec3 shadow_color_{ 0.13, 0, 0 };
    aten::vec3 midtone_color_{ 0.19, 0.09, 0.27 };
    aten::vec3 highlight_color_{ 0.13, 0.28, 0.55 };

    // Softness of transition from rim highlight to no highlights area.
    float rim_softness_{ 0.05F };

    // Global rim reflection contribution.
    float rim_weight_{ 1.0F };

    // Global rim reflection tint.
    aten::vec3 rim_tint_{ 1, 1, 1 };

    // Specular highlight to no highlights transition threshold.
    float glossy_level_{ 0.25F };

    // Softness of transition from specular highlight to no highlights area.
    float glossy_softness_{ 0.05F };

    aten::vec3 glossy_color_{ 1.0F, 0.55F, 0.2F };

    // Global specular reflection tint.
    aten::vec3 specular_tint_{ 1, 1, 1 };
};
