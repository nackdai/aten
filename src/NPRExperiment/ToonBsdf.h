#pragma once

#include <vector>

#include "NPRModule.h"

class ToonBsdf : public NPRModule {
public:
    ToonBsdf();
    ~ToonBsdf() = default;

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
    static constexpr int32_t RampInputEntryNum = 3;
    std::array<float, RampInputEntryNum> ramp_inputs_;
    std::array<aten::vec3, RampInputEntryNum> ramp_colors_;
    int32_t ramp_interp_type_{ 0 };

    bool enable_specular_{ false };
    float stretch_u_{ 1.0F };
    float stretch_v_{ 1.0F };
    float roughness_{ 0.0F };
};
