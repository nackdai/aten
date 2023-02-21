#pragma once

#include "aten.h"

// NOTE
// https://dl.acm.org/doi/pdf/10.1145/3095140.3095176

class MeasureEffectiveRetroreflectiveArea {
public:
    MeasureEffectiveRetroreflectiveArea() = default;
    ~MeasureEffectiveRetroreflectiveArea() = default;

public:
    // èâä˙âª.
    bool init(
        int32_t width, int32_t height,
        const char* pathVS,
        const char* pathFS);

    void draw(
        const aten::context& ctxt,
        const aten::camera* cam);

    aten::vec3 GenRay(real theta, real phi);

    real HitTest(real theta, real phi);

private:
    aten::shader m_shader;
    aten::GeomVertexBuffer m_vb;
    aten::GeomIndexBuffer m_ib;

    aten::GeomVertexBuffer m_vb_pts;
    aten::GeomIndexBuffer m_ib_pts;

    int32_t m_width{ 0 };
    int32_t m_height{ 0 };

    std::vector<aten::vec4> ray_orgs_;

    static constexpr int32_t RayOrgNum = 100;
    static constexpr size_t VtxNum = 6;
    static const std::array<aten::vec4, VtxNum> TriangleVtxs;
};
