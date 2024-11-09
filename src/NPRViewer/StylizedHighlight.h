#pragma once

#include <vector>

#include "aten.h"

class StylizedHighlight {
public:
    StylizedHighlight() = default;
    ~StylizedHighlight() = default;

    void Init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS);

    void UpdateHalfVectors(
        float translation,
        float scale,
        float split);

    void Draw(
        const aten::context& ctxt,
        const aten::Camera& cam);

private:
    std::vector<aten::vec3> points_;
    std::vector<aten::vec3> normals_;
    std::vector<aten::vec3> tangents_;

    aten::GeomVertexBuffer vb_normals_;
    aten::GeomVertexBuffer vb_halfs_;
    aten::GeomVertexBuffer vb_tangents_;
    aten::GeomIndexBuffer ib_;

    aten::shader shader_;
};
