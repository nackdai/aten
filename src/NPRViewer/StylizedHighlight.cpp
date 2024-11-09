#include <numeric>

#include <imgui.h>

#include "StylizedHighlight.h"
#include "visualizer/atengl.h"

namespace _detail {
    const std::array attribs = {
            aten::VertexAttrib{ GL_FLOAT, 3, sizeof(GLfloat), 0 },
    };
}

void StylizedHighlight::InitDebugVisual(
    int32_t width, int32_t height,
    std::string_view pathVS,
    std::string_view pathFS)
{
    shader_.init(width, height, pathVS, pathFS);

    constexpr float MaxAngle = aten::Deg2Rad(45);
    constexpr float MinAngle = aten::Deg2Rad(-45);
    constexpr float Step = 5;
    constexpr float StepAngle = (MaxAngle - MinAngle) / Step;

    // Radius of sphere mesh.
    constexpr float radii = 2.546F;

    std::vector<aten::vec4> nml_vtx;
    std::vector<aten::vec4> tangent_vtx;

    for (int32_t i = 0; i < Step; i++) {
        const auto theta = MinAngle + StepAngle * i;

        aten::vec3 nml(0.0F, 0.0F, 1.0F);

        aten::mat4 rot;
        rot.asRotateByX(theta);

        nml = rot.apply(nml);
        nml = normalize(nml);

        normals_.emplace_back(nml);

        aten::vec3 t, b;
        std::tie(t, b) = aten::GetTangentCoordinate(nml);

        tangents_.emplace_back(t);

        aten::vec3 point(nml);
        point *= radii;

        points_.emplace_back(point);

        tangent_vtx.emplace_back(aten::vec4(point, 1.0F));
        tangent_vtx.emplace_back(aten::vec4(point + t, 1.0F));

        nml_vtx.emplace_back(aten::vec4(point, 1.0F));
        nml_vtx.emplace_back(aten::vec4(point + nml, 1.0F));
    }

    // Vertex buffer.
    vb_normals_.init(
        sizeof(aten::vec4),
        nml_vtx.size(),
        0,
        _detail::attribs.data(),
        _detail::attribs.size(),
        nml_vtx.data());

    vb_tangents_.init(
        sizeof(aten::vec4),
        tangent_vtx.size(),
        0,
        _detail::attribs.data(),
        _detail::attribs.size(),
        tangent_vtx.data());

    UpdateHalfVectors();

    // Index buffer.
    std::vector<uint32_t> indices;
    indices.resize(nml_vtx.size());
    std::iota(indices.begin(), indices.end(), 0);
    ib_.init(indices.size(), indices.data());
}

void StylizedHighlight::PreRender(aten::shader& shader)
{
    auto h_translation_dt = shader.getHandle("translation_dt");
    CALL_GL_API(::glUniform1f(h_translation_dt, half_trans_t_));

    auto h_scale_t = shader.getHandle("scale_t");
    CALL_GL_API(::glUniform1f(h_scale_t, half_scale_));

    auto h_split_t = shader.getHandle("split_t");
    CALL_GL_API(::glUniform1f(h_split_t, half_split_t_));
}

void StylizedHighlight::UpdateHalfVectors()
{
    const aten::vec3 target_light(0.0F, 0.0F, 5.0F);
    const aten::vec3 eye(0.0F, 0.0F, 5.0F);

    std::vector<aten::vec4> half_vtx;

    for (int32_t i = 0; i < points_.size(); i++) {
        const auto& p = points_[i];
        const auto wi = eye - p;
        const auto wo = target_light - p;

        auto half = normalize(wi + wo);

        const auto& t = tangents_[i];

        // Translation.
        half = half + half_trans_t_ * t;
        half = normalize(half);

        // Directional scale.
        half = half - half_scale_ * dot(half, t) * t;
        half = normalize(half);

        // Split.
        auto sign_t = dot(half, t);
        sign_t = sign_t > 0
            ? 1
            : sign_t < 0 ? -1 : sign_t;
        half = half - half_split_t_ * sign_t * t;
        half = normalize(half);

        half_vtx.emplace_back(aten::vec4(p, 1.0F));
        half_vtx.emplace_back(aten::vec4(p + half, 1.0F));
    }

    if (vb_halfs_.isInitialized()) {
        vb_halfs_.update(half_vtx.size(), half_vtx.data());
    }
    else {
        vb_halfs_.init(
            sizeof(aten::vec4),
            half_vtx.size(),
            0,
            _detail::attribs.data(),
            _detail::attribs.size(),
            half_vtx.data());
    }
}

void StylizedHighlight::DrawDebugVisual(
    const aten::context& ctxt,
    const aten::Camera& cam)
{
    shader_.prepareRender(nullptr, false);

    auto camparam = cam.param();

    // TODO
    camparam.znear = float(0.1);
    camparam.zfar = float(10000.0);

    aten::mat4 mtx_W2V;
    aten::mat4 mtx_V2C;

    aten::mat4 mtx_L2W;

    mtx_W2V.lookat(
        camparam.origin,
        camparam.center,
        camparam.up);

    mtx_V2C.perspective(
        camparam.znear,
        camparam.zfar,
        camparam.vfov,
        camparam.aspect);

    aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

    auto hMtxW2C = shader_.getHandle("mtx_W2C");
    CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtx_W2C.a[0]));

    auto hMtxL2W = shader_.getHandle("mtx_L2W");
    auto hColor = shader_.getHandle("color");

    CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, mtx_L2W.data()));

    const auto line_vtx_num = normals_.size() * 2;

    // Normal.
    CALL_GL_API(::glUniform4f(hColor, 0.0f, 0.0f, 1.0f, 1.0F));
    ib_.draw(vb_normals_, aten::Primitive::Lines, 0, line_vtx_num);

    // Tangent.
    CALL_GL_API(::glUniform4f(hColor, 1.0f, 0.0f, 0.0f, 1.0F));
    ib_.draw(vb_tangents_, aten::Primitive::Lines, 0, line_vtx_num);

    // Half.
    CALL_GL_API(::glUniform4f(hColor, 0.0f, 1.0f, 0.0f, 1.0F));
    ib_.draw(vb_halfs_, aten::Primitive::Lines, 0, line_vtx_num);
}

void StylizedHighlight::EditParameter()
{
    ImGui::SliderFloat("trans t", &half_trans_t_, -1, 1);
    ImGui::SliderFloat("scale t", &half_scale_, 0.0, 1);
    ImGui::SliderFloat("split t", &half_split_t_, 0.0, 1);

    UpdateHalfVectors();
}
