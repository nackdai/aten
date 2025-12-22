#pragma once

#include <vector>

#include "NPRModule.h"

class StylizedHighlight : public NPRModule {
public:
    StylizedHighlight() = default;
    ~StylizedHighlight() = default;

    void InitDebugVisual(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS) override final;

    void PreRender(aten::shader& shader, const aten::PinholeCamera& camera) override final;

    void DrawDebugVisual(
        const aten::context& ctxt,
        const aten::Camera& cam) override final;

    void EditParameter() override final;

private:
    void UpdateHalfVectors();

    std::vector<aten::vec3> points_;
    std::vector<aten::vec3> normals_;
    std::vector<aten::vec3> tangents_;

    aten::GeomVertexBuffer vb_normals_;
    aten::GeomVertexBuffer vb_halfs_;
    aten::GeomVertexBuffer vb_tangents_;
    aten::GeomIndexBuffer ib_;

    aten::shader shader_;

    float half_trans_t_{ 0.0F };
    float half_trans_b_{ 0.0F };

    float half_scale_t_{ 0.0F };
    float half_scale_b_{ 0.0F };

    float half_split_t_{ 0.0F };
    float half_split_b_{ 0.0F };

    float half_square_sharp_{ 0.0F };
    float half_square_magnitude_{ 0.0F };
};
