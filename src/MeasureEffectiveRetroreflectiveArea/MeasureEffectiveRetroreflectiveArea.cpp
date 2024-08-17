#include <array>
#include <numeric>

#include "MeasureEffectiveRetroreflectiveArea.h"
#include "visualizer/atengl.h"
#include "math/intersect.h"

#pragma optimize( "", off)

// NOTE:
// Compute overlap area of two triangles (front and back) based on the ray direction.
// Ray direction means theta and phi.
// Generate ray based on the specified theta and phi,
// and then check if the generated ray hits the both triangles.
// Count the number of rays which hit the both triangles.
// Finally, compute the rates of hit rays.
// i.e. rate = <number of hit rays for both triangles> / <all number of rays>
// The origin position of all rays locates on the surface of the front triangle.
// So, <all number of rays> means the "rays on the front triangle".

namespace _detail {
    const float Pos = 1.0F;

    const std::array TriangleVtxs = {
        // Front face.
        aten::vec4(0,    Pos,    0, 1),
        aten::vec4(0,      0,  Pos, 1),
        aten::vec4(Pos,    0,    0, 1),
        // Back face.
        aten::vec4(-Pos,    0,    0, 1),
        aten::vec4(0, -Pos,    0, 1),
        aten::vec4(0,    0, -Pos, 1),
    };
}

void MeasureEffectiveRetroreflectiveArea::Init()
{
    if (!ray_orgs_.empty()) {
        return;
    }

    // Compute the origin position of rays.
    // The origin position of rays locate on the surface of the front triangles.

    // Obtain the vertices of the front triangle.
    const auto p0 = _detail::TriangleVtxs[0];
    const auto p1 = _detail::TriangleVtxs[1];
    const auto p2 = _detail::TriangleVtxs[2];

    const auto v0 = p1 - p0;
    const auto v1 = p2 - p0;

#if 0
    const aten::vec4 RectLeftBottom = p1;
    const aten::vec4 DirX = p2 - p1;

    const auto mid = float(0.5) * (p1 + p2);
    const aten::vec4 DirY = p0 - mid;

    constexpr int32_t RayOrgNum = 100;
    constexpr float Step = 1.0f / RayOrgNum;

    ray_orgs_.reserve(RayOrgNum * RayOrgNum);
    {
        for (int32_t y = 0; y <= RayOrgNum; y++) {
            const auto pos_y = Step * y;
            for (int32_t x = 0; x <= RayOrgNum; x++) {
                const auto pos_x = Step * x;

                const auto p = RectLeftBottom + pos_x * DirX + pos_y * DirY;
                ray_orgs_.push_back(p);
            }
        }
    }
#else
    constexpr auto step = 1.0F / RayOrgNum;
    ray_orgs_.reserve(RayOrgNum* RayOrgNum);

    for (int32_t y = 0; y <= RayOrgNum; y++) {
        const auto a = aten::saturate(y * step);

        for (int32_t x = 0; x <= RayOrgNum; x++) {
            const auto b = aten::saturate(x * step);

            if (a + b > 1.0F) {
                break;
            }

            const auto p = p0 + v0 * a + v1 * b;
            ray_orgs_.emplace_back(p);
        }
    }
#endif
}

bool MeasureEffectiveRetroreflectiveArea::InitForDebugVisualizing(
    int32_t width, int32_t height,
    std::string_view pathVS,
    std::string_view pathFS)
{
    constexpr std::array TriangleIdx = {
        0, 1, 2,
        3, 4, 5,
    };

    static const std::array attribs = {
        aten::VertexAttrib{ GL_FLOAT, 3, sizeof(GLfloat), 0 },
    };

    width_ = width;
    height_ = height;

    // vertex buffer.
    vertex_buffer_.init(
        sizeof(aten::vec4),
        _detail::TriangleVtxs.size(),
        0,
        attribs.data(),
        attribs.size(),
        _detail::TriangleVtxs.data());

    // index buffer.
    m_ib.init(TriangleIdx.size(), TriangleIdx.data());

    Init();

#if 1
    // The arguments to pass for GenRay are just for debug visibility.
    auto ray = GenRay(aten::Deg2Rad(35.1), aten::Deg2Rad(270));
    std::vector<aten::vec4> ps;

    for (const auto& org : ray_orgs_) {
        ps.push_back(org);
        ps.push_back(org + aten::vec4(ray.x, ray.y, ray.z, 1) * 10);
    }

    m_vb_pts.init(
        sizeof(aten::vec4),
        ps.size(),
        0,
        attribs.data(),
        attribs.size(),
        ps.data());

    std::vector<uint32_t> point_indices;
    point_indices.resize(ps.size());
    std::iota(point_indices.begin(), point_indices.end(), 0);
    m_ib_pts.init(point_indices.size(), point_indices.data());
#endif

    // Axis
    std::array<aten::vec4, 2> axis;

    constexpr std::array axis_idx = { 0, 1 };
    ib_axis_.init(axis_idx.size(), axis_idx.data());

    // X
    axis[0].set(0, 0, 0, 1);
    axis[1].set(10, 0, 0, 1);
    vb_axis_[0].init(
        sizeof(decltype(axis)::value_type), axis.size(), 0,
        attribs.data(), attribs.size(),
        axis.data());

    // Y
    axis[1].set(0, 10, 0, 1);
    vb_axis_[1].init(
        sizeof(decltype(axis)::value_type), axis.size(), 0,
        attribs.data(), attribs.size(),
        axis.data());

    axis[1].set(0, 0, 10, 1);
    vb_axis_[2].init(
        sizeof(decltype(axis)::value_type), axis.size(), 0,
        attribs.data(), attribs.size(),
        axis.data());

    return m_shader.init(width, height, pathVS, pathFS);
}

aten::vec3 MeasureEffectiveRetroreflectiveArea::GenRay(float theta, float phi)
{
    auto p0 = _detail::TriangleVtxs[0];
    auto p1 = _detail::TriangleVtxs[1];
    auto p2 = _detail::TriangleVtxs[2];

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto len = v0.length();

    v0 = normalize(v0);
    v1 = normalize(v1);

    auto n = static_cast<aten::vec3>(normalize(cross(v0, v1)));

    // NOTE
    // Inverse towards triangle plane.
    n = -n;

    //auto t = aten::getOrthoVector(n);
    //auto b = cross(n, t);

    auto t = normalize(aten::vec3(-0.5, 1, -0.5));
    auto b = normalize(aten::vec3(-1, 0, 1));

    //theta = std::clamp(theta, ThetaMin, ThetaMax);
    //phi = std::clamp(phi, PhiMin, PhiMax);

    const auto cos_theta = aten::cos(theta);
    const auto sin_theta = aten::sin(theta);

    const auto cos_phi = aten::cos(phi);
    const auto sin_phi = aten::sin(phi);

    auto x = sin_theta * cos_phi;
    auto y = sin_theta * sin_phi;
    auto z = cos_theta;

    auto d = x * t + y * b + z * n;
    d = normalize(d);
    return d;
}

float MeasureEffectiveRetroreflectiveArea::HitTest(float theta, float phi)
{
    auto d = GenRay(theta, phi);

    size_t front_face_hit_cnt = 0;
    size_t both_face_hit_cnt = 0;

    for (const auto& org : ray_orgs_) {
        aten::ray ray(org, d);

        const auto hit_res_front_face = aten::intersectTriangle(
            ray,
            _detail::TriangleVtxs[0], _detail::TriangleVtxs[1], _detail::TriangleVtxs[2]);

        const auto hit_res_back_face = aten::intersectTriangle(
            ray,
            _detail::TriangleVtxs[3], _detail::TriangleVtxs[4], _detail::TriangleVtxs[5]);

        if (hit_res_front_face.isIntersect) {
            front_face_hit_cnt++;

            if (hit_res_back_face.isIntersect) {
                both_face_hit_cnt++;
            }
        }
    }

    const auto hit_rate = front_face_hit_cnt > 0
        ? both_face_hit_cnt / static_cast<float>(front_face_hit_cnt)
        : 0.0F;

    return hit_rate;
}

void MeasureEffectiveRetroreflectiveArea::VisualizeForDebug(
    const aten::context& ctxt,
    const aten::Camera* cam)
{
    m_shader.prepareRender(nullptr, false);

    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
        1.0f,
        0);

    CALL_GL_API(::glEnable(GL_DEPTH_TEST));
    //CALL_GL_API(::glEnable(GL_CULL_FACE));
    CALL_GL_API(::glDisable(GL_CULL_FACE));

    auto camparam = cam->param();

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

    auto hMtxW2C = m_shader.getHandle("mtx_W2C");
    CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtx_W2C.a[0]));

    auto hMtxL2W = m_shader.getHandle("mtx_L2W");
    auto hColor = m_shader.getHandle("color");
    auto hNormal = m_shader.getHandle("normal");

    // NOTE
    // Just only one triangle
    static constexpr size_t PrimCnt = 2;

    CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, mtx_L2W.data()));
    CALL_GL_API(::glUniform3f(hNormal, 1.0f, 0.0f, 0.0f));

    CALL_GL_API(::glUniform3f(hColor, 1.0f, 0.0f, 0.0f));
    m_ib.draw(vertex_buffer_, aten::Primitive::Triangles, 0, 1);

    CALL_GL_API(::glUniform3f(hColor, 0.0f, 1.0f, 0.0f));
    m_ib.draw(vertex_buffer_, aten::Primitive::Triangles, 3, 1);

    CALL_GL_API(::glUniform3f(hColor, 0.0f, 0.0f, 0.0f));
    m_ib_pts.draw(m_vb_pts, aten::Primitive::Lines, 0, ray_orgs_.size() * 2);

    // Axis

    // X
    CALL_GL_API(::glUniform3f(hColor, 1.0f, 0.0f, 0.0f));
    ib_axis_.draw(vb_axis_[0], aten::Primitive::Lines, 0, 2);
    // Y
    CALL_GL_API(::glUniform3f(hColor, 0.0f, 1.0f, 0.0f));
    ib_axis_.draw(vb_axis_[1], aten::Primitive::Lines, 0, 2);
    // Z
    CALL_GL_API(::glUniform3f(hColor, 0.0f, 0.0f, 1.0f));
    ib_axis_.draw(vb_axis_[2], aten::Primitive::Lines, 0, 2);
}
