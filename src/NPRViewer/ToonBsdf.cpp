#include <numeric>

#include <imgui.h>

#include "ToonBsdf.h"
#include "visualizer/atengl.h"

ToonBsdf::ToonBsdf()
    : NPRModule()
    , ramp_inputs_{ 0.0F, 0.5F, 1.0F }
    , ramp_colors_{ aten::vec3(0.0F, 0.0F, 0.0F), aten::vec3(0.5F, 0.5F, 0.5F), aten::vec3(1.0F, 1.0F, 1.0F) }
{
}

void ToonBsdf::PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
{
    const auto& cam_param = camera.param();
    auto h_eye = shader.getHandle("eye");
    CALL_GL_API(::glUniform3f(h_eye, cam_param.origin.x, cam_param.origin.y, cam_param.origin.z));

    auto h_ramp_inputs = shader.getHandle("ramp_inputs");
    CALL_GL_API(::glUniform1fv(h_ramp_inputs, RampInputEntryNum, ramp_inputs_.data()));

    auto h_ramp_colors = shader.getHandle("ramp_colors");
    CALL_GL_API(::glUniform3fv(h_ramp_colors, RampInputEntryNum, &ramp_colors_[0].x));

    auto h_ramp_interp_type = shader.getHandle("ramp_interp_type");
    CALL_GL_API(::glUniform1i(h_ramp_interp_type, ramp_interp_type_));

    auto h_stretch_u = shader.getHandle("StretchU");
    if (h_stretch_u >= 0)
    {
        enable_specular_ = true;

        CALL_GL_API(::glUniform1f(h_stretch_u, stretch_u_));

        auto h_stretch_v = shader.getHandle("StretchV");
        CALL_GL_API(::glUniform1f(h_stretch_u, stretch_v_));

        auto h_roughness = shader.getHandle("Roughness");
        CALL_GL_API(::glUniform1f(h_roughness, roughness_));
    }
}

void ToonBsdf::EditParameter()
{
    ImGui::SliderInt("ramp_interp_type", &ramp_interp_type_, 0, 3);

    for (int32_t i = 0; i < RampInputEntryNum; i++)
    {
        ImGui::SliderFloat(("ramp_input_" + std::to_string(i)).c_str(), &ramp_inputs_[i], 0.0F, 1.0F);
        ImGui::ColorEdit3(("ramp_color_" + std::to_string(i)).c_str(), &ramp_colors_[i].x);
    }

    std::sort(
        ramp_inputs_.begin(),
        ramp_inputs_.end());

    if (enable_specular_) {
        if (ImGui::CollapsingHeader("Specular")) {
            ImGui::SliderFloat("stretch_u", &stretch_u_, 0.0F, 1.0F);
            ImGui::SliderFloat("stretch_v", &stretch_v_, 0.0F, 1.0F);
            ImGui::SliderFloat("roughness", &roughness_, 0.0F, 1.0F);
        }
    }
}
