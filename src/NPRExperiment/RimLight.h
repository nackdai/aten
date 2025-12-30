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

///////////////////////////////////////////////////

class RimLight_v2 : public NPRModule {
public:
    RimLight_v2() = default;
    ~RimLight_v2() = default;

    void InitDebugVisual(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS) override final
    {
    }

    void PreRender(aten::shader& shader, const aten::PinholeCamera& camera) override final;

    void DrawDebugVisual(
        const aten::context& ctxt,
        const aten::Camera& cam)
    {
    }

    void EditParameter() override final;

private:
    aten::vec3 light_pos;
    float rim_offset{ 0.5F };
    float front_rim_intensity{ 0.75f };
    aten::vec3 front_rim_color{ 1.0F, 0.52211F, 0.037902F };
    float back_rim_offset{ 0.005f };
    float back_rim_intensity{ 0.75f };
    aten::vec3 back_rim_color{ 0.114791F, 0.0F, 1.0F };

    float rim_width{ 1.469821F };
    float rim_intensity{ 8.120095F };
    float rim_strength{ 3000.0F };
    float rim_contrast{ 10.0F };
};
