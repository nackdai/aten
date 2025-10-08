#include <numeric>

#include <imgui.h>

#include "RimLight.h"
#include "visualizer/atengl.h"


void RimLight::PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
{
    // TODO;
    rim_light_pos_ = aten::vec3(0, 0, 20);
    auto h_light_pos = shader.GetHandle("light_pos");
    CALL_GL_API(::glUniform3f(h_light_pos, rim_light_pos_.x, rim_light_pos_.y, rim_light_pos_.z));

    auto h_light_color = shader.GetHandle("light_color");
    CALL_GL_API(::glUniform3f(h_light_color, rim_light_color_.x, rim_light_color_.y, rim_light_color_.z));

    const auto& cam_param = camera.param();
    auto h_eye = shader.GetHandle("eye");
    CALL_GL_API(::glUniform3f(h_eye, cam_param.origin.x, cam_param.origin.y, cam_param.origin.z));

    auto h_width = shader.GetHandle("width");
    CALL_GL_API(::glUniform1f(h_width, width_));

    auto h_softness = shader.GetHandle("softness");
    CALL_GL_API(::glUniform1f(h_softness, softness_));

    auto h_spread = shader.GetHandle("spread");
    CALL_GL_API(::glUniform1f(h_spread, spread_));
}

void RimLight::EditParameter()
{
    std::array f = { rim_light_color_.x, rim_light_color_.y, rim_light_color_.z };
    ImGui::ColorEdit3("Rim Light Color", f.data());

    rim_light_color_.x = f[0];
    rim_light_color_.y = f[1];
    rim_light_color_.z = f[2];

    ImGui::SliderFloat("width", &width_, 0, 1);
    ImGui::SliderFloat("softness", &softness_, 0.0, 1);
    ImGui::SliderFloat("spread", &spread_, 0.0, 1);
}
