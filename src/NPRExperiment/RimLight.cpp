#include <numeric>

#include <imgui.h>

#include "RimLight.h"
#include "visualizer/atengl.h"

void RimLight::PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
{
    // TODO;
    rim_light_pos_ = aten::vec3(0, 0, 20);
    auto h_light_pos = shader.getHandle("light_pos");
    CALL_GL_API(::glUniform3f(h_light_pos, rim_light_pos_.x, rim_light_pos_.y, rim_light_pos_.z));

    auto h_light_color = shader.getHandle("light_color");
    CALL_GL_API(::glUniform3f(h_light_color, rim_light_color_.x, rim_light_color_.y, rim_light_color_.z));

    const auto& cam_param = camera.param();
    auto h_eye = shader.getHandle("eye");
    CALL_GL_API(::glUniform3f(h_eye, cam_param.origin.x, cam_param.origin.y, cam_param.origin.z));

    auto h_width = shader.getHandle("width");
    CALL_GL_API(::glUniform1f(h_width, width_));

    auto h_softness = shader.getHandle("softness");
    CALL_GL_API(::glUniform1f(h_softness, softness_));

    auto h_spread = shader.getHandle("spread");
    CALL_GL_API(::glUniform1f(h_spread, spread_));
}

void RimLight::EditParameter()
{
    ImGui::ColorEdit3("Rim Light Color", &rim_light_color_.x);
    ImGui::SliderFloat("width", &width_, 0, 1);
    ImGui::SliderFloat("softness", &softness_, 0.0, 1);
    ImGui::SliderFloat("spread", &spread_, 0.0, 1);
}

///////////////////////////////////////////////////

void RimLight_v2::PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
{
    light_pos = aten::vec3(0, 0, 20);
    auto h_light_pos = shader.getHandle("light_pos");
    CALL_GL_API(::glUniform3f(h_light_pos, light_pos.x, light_pos.y, light_pos.z));

    const auto& cam_param = camera.param();
    auto h_eye = shader.getHandle("eye");
    CALL_GL_API(::glUniform3f(h_eye, cam_param.origin.x, cam_param.origin.y, cam_param.origin.z));

    auto h_rim_offset = shader.getHandle("rim_offset");
    CALL_GL_API(::glUniform1f(h_rim_offset, rim_offset));

    auto h_front_rim_intensity = shader.getHandle("front_rim_intensity");
    CALL_GL_API(::glUniform1f(h_front_rim_intensity, front_rim_intensity));

    auto h_front_rim_color = shader.getHandle("front_rim_color");
    CALL_GL_API(::glUniform3f(h_front_rim_color, front_rim_color.x, front_rim_color.y, front_rim_color.z));

    auto h_back_rim_offset = shader.getHandle("back_rim_offset");
    CALL_GL_API(::glUniform1f(h_back_rim_offset, back_rim_offset));

    auto h_back_rim_intensity = shader.getHandle("back_rim_intensity");
    CALL_GL_API(::glUniform1f(h_back_rim_intensity, back_rim_intensity));

    auto h_back_rim_color = shader.getHandle("back_rim_color");
    CALL_GL_API(::glUniform3f(h_back_rim_color, back_rim_color.x, back_rim_color.y, back_rim_color.z));

    auto h_rim_width = shader.getHandle("rim_width");
    CALL_GL_API(::glUniform1f(h_rim_width, rim_width));

    auto h_rim_intensity = shader.getHandle("rim_intensity");
    CALL_GL_API(::glUniform1f(h_rim_intensity, rim_intensity));

    auto h_rim_strength = shader.getHandle("rim_strength");
    CALL_GL_API(::glUniform1f(h_rim_strength, rim_strength));

    auto h_rim_contrast = shader.getHandle("rim_contrast");
    CALL_GL_API(::glUniform1f(h_rim_contrast, rim_contrast));
}

void RimLight_v2::EditParameter()
{
    ImGui::SliderFloat("rim_offset", &rim_offset, 0, 1);
    ImGui::ColorEdit3("front_rim_color", &front_rim_color.x);
    ImGui::ColorEdit3("back_rim_color", &back_rim_color.x);
    ImGui::SliderFloat("front_rim_intensity", &front_rim_intensity, 0.0, 1);
    ImGui::SliderFloat("back_rim_offset", &back_rim_offset, 0, 1);
    ImGui::SliderFloat("back_rim_intensity", &back_rim_intensity, 0.0, 1);

    ImGui::SliderFloat("rim_width", &rim_width, 0.0, 10);
    ImGui::SliderFloat("rim_intensity", &rim_intensity, 0.0, 10);
    ImGui::SliderFloat("rim_strength", &rim_strength, 0, 10000);
    ImGui::SliderFloat("rim_contrast", &rim_contrast, 0, 100);
}
