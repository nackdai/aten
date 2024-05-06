#pragma once

#include "aten.h"

// NOTE
// https://dl.acm.org/doi/pdf/10.1145/3095140.3095176

class MeasureEffectiveRetroreflectiveArea {
public:
    MeasureEffectiveRetroreflectiveArea() = default;
    ~MeasureEffectiveRetroreflectiveArea() = default;

    static constexpr float ThetaMin = 0.0F;
    static constexpr float ThetaMax = AT_MATH_PI_HALF;

    static constexpr float PhiMin = 0;
    //static constexpr float PhiMax = AT_MATH_PI_2;
    static constexpr float PhiMax = AT_MATH_PI;

    static constexpr int32_t RayOrgNum = 100;

    void Init();

    bool InitForDebugVisualizing(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS);

    void VisualizeForDebug(
        const aten::context& ctxt,
        const aten::camera* cam);

    aten::vec3 GenRay(float theta, float phi);

    float HitTest(float theta, float phi);

    bool IsValid() const
    {
        return m_shader.IsValid();
    }

private:
    aten::shader m_shader;
    aten::GeomVertexBuffer vertex_buffer_;
    aten::GeomIndexBuffer m_ib;

    aten::GeomVertexBuffer m_vb_pts;
    aten::GeomIndexBuffer m_ib_pts;

    // Axis x/y/z.
    aten::GeomVertexBuffer vb_axis_[3];
    aten::GeomIndexBuffer ib_axis_;

    int32_t width_{ 0 };
    int32_t height_{ 0 };

    std::vector<aten::vec4> ray_orgs_;
};
