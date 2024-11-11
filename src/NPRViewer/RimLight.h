#pragma once

#include <vector>

#include "NPRModule.h"

class RimLight : public NPRModule {
public:
    RimLight() = default;
    ~RimLight() = default;

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
    aten::vec3 rim_light_color_{ aten::vec3(1, 1, 1) };
    aten::vec3 rim_light_pos_;
    float width_{ 0.0F };
    float softness_{ 0.0F };
    float spread_{ 0.0F };
};
