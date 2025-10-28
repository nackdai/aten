#include <numeric>

#include <imgui.h>

#include "NPRShading.h"
#include "visualizer/atengl.h"


void NPRShading::PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
{
    const auto& cam_param = camera.param();
    auto h_eye = shader.getHandle("eye");
    CALL_GL_API(::glUniform3f(h_eye, cam_param.origin.x, cam_param.origin.y, cam_param.origin.z));

    // Diffuse.
    auto h_diffuse_tint = shader.getHandle("diffuse_tint");
    CALL_GL_API(::glUniform3f(h_diffuse_tint, diffuse_tint_.x, diffuse_tint_.y, diffuse_tint_.z));

    auto h_shadow_color = shader.getHandle("shadow_color");
    CALL_GL_API(::glUniform3f(h_shadow_color, shadow_color_.x, shadow_color_.y, shadow_color_.z));

    auto h_midtone_color = shader.getHandle("midtone_color");
    CALL_GL_API(::glUniform3f(h_midtone_color, midtone_color_.x, midtone_color_.y, midtone_color_.z));

    auto h_highlight_color = shader.getHandle("highlight_color");
    CALL_GL_API(::glUniform3f(h_highlight_color, highlight_color_.x, highlight_color_.y, highlight_color_.z));

    auto h_shadow_level = shader.getHandle("shadow_level");
    CALL_GL_API(::glUniform1f(h_shadow_level, shadow_level_));

    auto h_midtone_level = shader.getHandle("midtone_level");
    CALL_GL_API(::glUniform1f(h_midtone_level, midtone_level_));

    auto h_diffuse_softness = shader.getHandle("diffuse_softness");
    CALL_GL_API(::glUniform1f(h_diffuse_softness, diffuse_softness_));

    // Rim.
    auto h_rim_tint = shader.getHandle("rim_tint");
    CALL_GL_API(::glUniform3f(h_rim_tint, rim_tint_.x, rim_tint_.y, rim_tint_.z));

    auto h_rim_weight = shader.getHandle("rim_weight");
    CALL_GL_API(::glUniform1f(h_rim_weight, rim_weight_));

    auto h_rim_softness = shader.getHandle("rim_softness");
    CALL_GL_API(::glUniform1f(h_rim_softness, rim_softness_));

    // Specular.
    auto h_glossy_color = shader.getHandle("glossy_color");
    CALL_GL_API(::glUniform3f(h_glossy_color, glossy_color_.x, glossy_color_.y, glossy_color_.z));

    auto h_specular_tint = shader.getHandle("specular_tint");
    CALL_GL_API(::glUniform3f(h_specular_tint, specular_tint_.x, specular_tint_.y, specular_tint_.z));

    auto h_glossy_level = shader.getHandle("glossy_level");
    CALL_GL_API(::glUniform1f(h_glossy_level, glossy_level_));

    auto h_glossy_softness = shader.getHandle("glossy_softness");
    CALL_GL_API(::glUniform1f(h_glossy_softness, glossy_softness_));
}

void NPRShading::EditParameter()
{
    if (ImGui::CollapsingHeader("Diffuse"))
    {
        ImGui::SliderFloat("shadow_level", &shadow_level_, 0.0F, 1.0F);
        ImGui::SliderFloat("midtone_level", &midtone_level_, 0.0F, 1.0F);
        ImGui::SliderFloat("diffuse_softness", &diffuse_softness_, 0.001F, 0.5F);
        ImGui::ColorEdit3("diffuse_tint", &diffuse_tint_.x);
        ImGui::ColorEdit3("shadow_color", &shadow_color_.x);
        ImGui::ColorEdit3("midtone_color", &midtone_color_.x);
        ImGui::ColorEdit3("highlight_color", &highlight_color_.x);
    }

    if (ImGui::CollapsingHeader("Rim"))
    {
        ImGui::SliderFloat("rim_softness", &rim_softness_, 0.001F, 1.0F);
        ImGui::SliderFloat("rim_weight", &rim_weight_, 0.0F, 1.0F);
        ImGui::ColorEdit3("rim_tint", &rim_tint_.x);
    }

    if (ImGui::CollapsingHeader("Specular"))
    {
        ImGui::SliderFloat("glossy_level", &glossy_level_, 0.0F, 1.0F);
        ImGui::SliderFloat("glossy_softness", &glossy_softness_, 0.001F, 1.0F);
        ImGui::ColorEdit3("glossy_color", &glossy_color_.x);
        ImGui::ColorEdit3("specular_tint", &specular_tint_.x);
    }
}
