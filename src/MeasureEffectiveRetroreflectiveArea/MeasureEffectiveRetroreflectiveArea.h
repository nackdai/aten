#pragma once

#include "aten.h"

class MeasureEffectiveRetroreflectiveArea {
public:
    MeasureEffectiveRetroreflectiveArea() = default;
    ~MeasureEffectiveRetroreflectiveArea() = default;

public:
    // èâä˙âª.
    bool init(
        int width, int height,
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

    int m_width{ 0 };
    int m_height{ 0 };

    std::vector<aten::vec4> ray_orgs_;

    static constexpr int32_t RayOrgNum = 100;
    static constexpr size_t VtxNum = 6;
    static const std::array<aten::vec4, VtxNum> TriangleVtxs;
};
