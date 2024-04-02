#include <array>
#include <numeric>

#include "MeasureEffectiveRetroreflectiveArea.h"
#include "visualizer/atengl.h"
#include "math/intersect.h"

static const float Pos = 1.0f / aten::sqrt(2);

const std::array<aten::vec4, MeasureEffectiveRetroreflectiveArea::VtxNum>
MeasureEffectiveRetroreflectiveArea::TriangleVtxs = {
    // Front face.
    aten::vec4(   0,  Pos,    0, 1),
    aten::vec4(   0,    0,  Pos, 1),
    aten::vec4( Pos,    0,    0, 1),
    // Back face.
    aten::vec4(-Pos,    0,    0, 1),
    aten::vec4(   0, -Pos,    0, 1),
    aten::vec4(   0,    0, -Pos, 1),
};

void MeasureEffectiveRetroreflectiveArea::Init()
{
    if (!ray_orgs_.empty()) {
        return;
    }

    auto p0 = TriangleVtxs[0];
    auto p1 = TriangleVtxs[1];
    auto p2 = TriangleVtxs[2];

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto len = v0.length();

    v0 = normalize(v0);
    v1 = normalize(v1);

    auto n = normalize(cross(v0, v1));

    const aten::vec4 RectLeftBottom = p1;
    const aten::vec4 DirX = p2 - p1;

    const auto mid = real(0.5) * (p1 + p2);
    const aten::vec4 DirY = p0 - mid;

    ray_orgs_.reserve(RayOrgNum * RayOrgNum);
    {
        static constexpr float Step = 1.0f / RayOrgNum;

        for (int32_t y = 0; y < RayOrgNum; y++) {
            const auto pos_y = Step * y;
            for (int32_t x = 0; x < RayOrgNum; x++) {
                const auto pos_x = Step * x;

                const auto p = RectLeftBottom + pos_x * DirX + pos_y * DirY;
                ray_orgs_.push_back(p);
            }
        }
    }
}

bool MeasureEffectiveRetroreflectiveArea::InitDraw(
    int32_t width, int32_t height,
    std::string_view pathVS,
    std::string_view pathFS)
{
    static const std::array TriangleIdx = {
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
        TriangleVtxs.size(),
        0,
        attribs.data(),
        attribs.size(),
        TriangleVtxs.data());

    // index buffer.
    m_ib.init(TriangleIdx.size(), TriangleIdx.data());

    Init();

#if 1
    // The arguments to pass for GenRay are just for debug visibility.
    auto ray = GenRay(Deg2Rad(30), Deg2Rad(-162));
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

    return m_shader.init(width, height, pathVS, pathFS);
}

aten::vec3 MeasureEffectiveRetroreflectiveArea::GenRay(real theta, real phi)
{
    auto p0 = TriangleVtxs[0];
    auto p1 = TriangleVtxs[1];
    auto p2 = TriangleVtxs[2];

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto len = v0.length();

    v0 = normalize(v0);
    v1 = normalize(v1);

    auto n = static_cast<aten::vec3>(normalize(cross(v0, v1)));

    // NOTE
    // Inverse towards triangle plane.
    n = -n;

    auto t = aten::getOrthoVector(n);
    auto b = cross(n, t);

    theta = std::clamp(theta, real(0), real(AT_MATH_PI_HALF));
    phi = std::clamp(phi, -real(AT_MATH_PI), real(AT_MATH_PI));

    auto cos_theta = aten::cos(theta);
    auto sin_theta = aten::sin(theta);

    auto x = sin_theta * aten::cos(phi);
    auto y = sin_theta * aten::sin(phi);
    auto z = cos_theta;

    auto d = x * t + y * b + z * n;
    d = normalize(d);
    return d;
}

real MeasureEffectiveRetroreflectiveArea::HitTest(real theta, real phi)
{
    auto d = GenRay(theta, phi);

    size_t front_face_hit_cnt = 0;
    size_t both_face_hit_cnt = 0;

    for (const auto& org : ray_orgs_) {
        aten::ray ray(org, d);

        auto hit_res_front_face = aten::intersectTriangle(
            ray,
            TriangleVtxs[0], TriangleVtxs[1], TriangleVtxs[2]);

        auto hit_res_back_face = aten::intersectTriangle(
            ray,
            TriangleVtxs[3], TriangleVtxs[4], TriangleVtxs[5]);

        if (hit_res_front_face.isIntersect) {
            front_face_hit_cnt++;

            if (hit_res_back_face.isIntersect) {
                both_face_hit_cnt++;
            }
        }
    }

    real hit_rate = real(both_face_hit_cnt) / real(front_face_hit_cnt);

    return hit_rate;
}

void MeasureEffectiveRetroreflectiveArea::draw(
    const aten::context& ctxt,
    const aten::camera* cam)
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
    camparam.znear = real(0.1);
    camparam.zfar = real(10000.0);

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
    //m_ib_pts.draw(m_vb_pts, aten::Primitive::Points, 0, ray_orgs_.size());
    m_ib_pts.draw(m_vb_pts, aten::Primitive::Lines, 0, ray_orgs_.size() * 2);
}
