#pragma once

#include "aten.h"
#include "renderer/feature_line.h"

// NOTE
// This was test app to confirme sample ray, but current this doesn't work as expected.
// But, keep this app for a while.

class FeatureLineSampleRay {
public:
    FeatureLineSampleRay() = default;
    ~FeatureLineSampleRay() = default;

public:
    // èâä˙âª.
    bool init(
        int width, int height,
        const char* pathVS,
        const char* pathFS);

    void draw(
        const aten::context& ctxt,
        const aten::camera* cam);

private:
    void updateVtxBuffers(
        aten::FeatureLine::Disc* tmp_hrec,
        const aten::ray& query_ray,
        const aten::hitrecord& query_ray_hrec,
        const aten::ray& sample_ray,
        const aten::vec3& sample_ray_hit_pos);

    void draw(
        uint32_t count,
        const aten::camera* cam);

private:
    aten::shader shader_;
    aten::GeomVertexBuffer vb_plane_;
    aten::GeomIndexBuffer ib_plane_;

    aten::GeomVertexBuffer vb_query_ray_;
    aten::GeomVertexBuffer vb_sample_ray_;
    aten::GeomVertexBuffer vb_plane_nml_;
    aten::GeomIndexBuffer ib_line_;
};
