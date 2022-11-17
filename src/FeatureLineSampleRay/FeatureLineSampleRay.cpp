#include <array>
#include <numeric>

#include "FeatureLineSampleRay.h"
#include "visualizer/atengl.h"
#include "math/intersect.h"
#include "renderer/feature_line.h"

static constexpr real SceneScaleFactor = 10;
static constexpr size_t Num = 4;

static const std::array<aten::vec3, Num> ray_hit_points_ = {
    //aten::vec3(-1.00095677, 0.00000000, 0.57637328),
    //aten::vec3(-1.00600839, 0.00691744, 0.57426405),
    //aten::vec3(-0.99144155, 0.00000000, 0.58998346),
    aten::vec3(0.00683172792, 0.0105081657, 0.587637424),
    aten::vec3(-0.00405663252, 0.00000000, 0.585765839),
    aten::vec3(0.00336601585, 0.0271742288, 0.586561918),
    aten::vec3(-0.0169057846, 0.00000000, 0.625110209),
};

static const std::array<aten::vec3, Num> ray_dir_ = {
    aten::vec3(0.00262012682, -0.379490942, -0.925191760),
    aten::vec3(-0.713885546, -0.689247906, -0.123712361),
    aten::vec3(0.263539493, 0.964234471, 0.0282643028),
    aten::vec3(-0.394896090, -0.529474378, 0.750808954),
};

static const std::array<aten::vec3, Num> ray_hit_nmls_ = {
    //aten::vec3(0.00000000, 1.00000000, 0.00000000),
    //aten::vec3(0.99993736, 0.01005000, 0.00492600),
    //aten::vec3(0.00000000, 1.00000000, 0.00000000),
    aten::vec3(-0.296399117, 0.00000000, 0.955064297),
    aten::vec3(0.00000000, 1.00000000, 0.00000000),
    aten::vec3(-0.296399087, 0.00000000, 0.955064237),
    aten::vec3(0.00000000, 1.00000000, 0.00000000),
};

bool FeatureLineSampleRay::init(
    int width, int height,
    const char* pathVS,
    const char* pathFS)
{
    static const std::array<aten::vec4, 2> LineVtx = {
        aten::vec4(0, 0, 0, 1),
        aten::vec4(0, 1, 0, 1),
    };
    static const std::array<aten::vec4, 4> PlaneVtx = {
        aten::vec4(0, 0, 0, 1),
        aten::vec4(0, 0, 1, 1),
        aten::vec4(1, 0, 0, 1),
        aten::vec4(1, 0, 1, 1),
    };

    static const std::array<uint32_t, 2> LineIdx = {
        0, 1,
    };
    static const std::array<uint32_t, 6> PlaneIdx = {
        0, 1, 2,
        1, 3, 2,
    };

    static const std::array<aten::VertexAttrib, 1> attribs = {
        aten::VertexAttrib{ GL_FLOAT, 3, sizeof(GLfloat), 0 },
    };

    // Plane
    vb_plane_.init(
        sizeof(aten::vec4),
        PlaneVtx.size(),
        0,
        attribs.data(),
        attribs.size(),
        PlaneVtx.data(),
        true);
    ib_plane_.init(PlaneIdx.size(), PlaneIdx.data());

    // Query ray.
    vb_query_ray_.init(
        sizeof(aten::vec4),
        LineVtx.size(),
        0,
        attribs.data(),
        attribs.size(),
        LineVtx.data(),
        true);
    vb_sample_ray_.init(
        sizeof(aten::vec4),
        LineVtx.size(),
        0,
        attribs.data(),
        attribs.size(),
        LineVtx.data(),
        true);
    vb_plane_nml_.init(
        sizeof(aten::vec4),
        LineVtx.size(),
        0,
        attribs.data(),
        attribs.size(),
        LineVtx.data(),
        true);

    ib_line_.init(LineIdx.size(), LineIdx.data());

    return shader_.init(width, height, pathVS, pathFS);
}

void FeatureLineSampleRay::draw(
    const aten::context& ctxt,
    const aten::camera* cam)
{
    shader_.prepareRender(nullptr, false);

    CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
    CALL_GL_API(::glClearDepthf(1.0f));
    CALL_GL_API(::glClearStencil(0));
    CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    CALL_GL_API(::glEnable(GL_DEPTH_TEST));
    //CALL_GL_API(::glEnable(GL_CULL_FACE));
    CALL_GL_API(::glDisable(GL_CULL_FACE));

    const auto pixel_width = 0.00161802175f;

    constexpr real FeatureLineWidth = 1;

    aten::FeatureLine::SampleRayDesc sample_ray_desc;
    aten::FeatureLine::Disc prev_disc;
    aten::hitrecord prev_query_ray_hrec;

    class DummySampler : public aten::sampler {
    public:
        DummySampler() = default;
        virtual ~DummySampler() = default;
        virtual real nextSample() override final { return real(0); }

        virtual aten::vec2 nextSample2D() override final
        {
            // [0-1]
            //return aten::vec2(0.0534606539, 0.524640083);
            //return aten::vec2(0.776053011, 0.0637817159);
            //return aten::vec2(0.277246207, 0.971535325);
            return aten::vec2(0.102102391, 0.293894410);
        }
    };
    DummySampler sampler;

    aten::ray query_ray;
    query_ray.org = aten::vec3(0, 1, 3);
    //query_ray.dir = aten::vec3(-0.356669337, -0.356328368, -0.863606930);
    query_ray.dir = ray_dir_[0];

    auto disc = aten::FeatureLine::generateDisc(query_ray, FeatureLineWidth, pixel_width);
    auto sample_ray = aten::FeatureLine::generateSampleRay(sample_ray_desc, sampler, query_ray, disc);

    real accumulated_hit_point_distance = 1;

    aten::vec3 sample_ray_hit_pos;

    for (size_t i = 0; i < ray_hit_points_.size(); i++) {
    //for (size_t i = 0; i < 2; i++) {
        // Query ray generation.
        if (i > 0) {
            query_ray.org = ray_hit_points_[i - 1];
            auto dir = ray_hit_points_[i] - ray_hit_points_[i - 1];
            dir = normalize(dir);
            const auto d = dot(ray_dir_[i], dir);
            query_ray.dir = ray_dir_[i];
        }

        // Assume scene hit happens here.
        aten::hitrecord query_ray_hrec;
        query_ray_hrec.p = ray_hit_points_[i];
        query_ray_hrec.normal = ray_hit_nmls_[i];

        // disc.centerはquery_ray.orgに一致する.
        // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
        // accumulated_hit_point_distanceでカメラとdiscの距離がすでに含まれている.
        auto hit_point_distance = length(query_ray_hrec.p - disc.center);

        prev_disc = disc;
        disc = aten::FeatureLine::computeNextDisc(
            query_ray_hrec.p,
            query_ray.dir,
            prev_disc.radius,
            hit_point_distance,
            accumulated_hit_point_distance);

        aten::vec3 pos_on_disc;

        if (i > 0) {
            const auto res_next_sample_ray = aten::FeatureLine::computeNextSampleRay(
                sample_ray_desc,
                prev_disc, disc);
            AT_ASSERT(std::get<0>(res_next_sample_ray));
            sample_ray = std::get<1>(res_next_sample_ray);
            pos_on_disc = std::get<2>(res_next_sample_ray);
        }

        // For rendering.
        const auto plane = aten::FeatureLine::computePlane(query_ray_hrec);
        const auto res_sample_ray_hit = aten::FeatureLine::computeRayHitPosOnPlane(plane, sample_ray);
        //AT_ASSERT(std::get<0>(res_sample_ray_hit));
        sample_ray_hit_pos = std::get<1>(res_sample_ray_hit);

        updateVtxBuffers(
            //&disc,
            nullptr,
            query_ray,
            query_ray_hrec,
            sample_ray,
            //i == 0 ? sample_ray_hit_pos : pos_on_disc);
            sample_ray_hit_pos);

        draw(i, cam);

        prev_query_ray_hrec = query_ray_hrec;
        accumulated_hit_point_distance += hit_point_distance;
    }
}

namespace {
    void computePlaneVtx(
        std::array<aten::vec4, 4>& vtxs,
        const aten::vec3& p, const aten::vec3& nml,
        real scale)
    {
        std::array<aten::vec4, 4> plane_vtx_base = {
            aten::vec4(-scale,  scale, 0, 1),
            aten::vec4(-scale, -scale, 0, 1),
            aten::vec4( scale,  scale, 0, 1),
            aten::vec4( scale, -scale, 0, 1),
        };

        const auto n = nml;
        const auto t = aten::getOrthoVector(n);
        const auto b = cross(n, t);

        aten::mat4 mtx_axes(t, b, n);
        for (size_t i = 0; i < plane_vtx_base.size(); i++) {
            vtxs[i] = mtx_axes.applyXYZ(plane_vtx_base[i]);
            vtxs[i] += aten::vec4(p, 0);
        }
    }
}

void FeatureLineSampleRay::updateVtxBuffers(
    aten::FeatureLine::Disc* tmp_hrec,
    const aten::ray& query_ray,
    const aten::hitrecord& query_ray_hrec,
    const aten::ray& sample_ray,
    const aten::vec3& sample_ray_hit_pos)
{
    // Update Plane
    std::array<aten::vec4, 4> plane_vtxs;
    if (tmp_hrec) {
        computePlaneVtx(plane_vtxs, tmp_hrec->center, tmp_hrec->normal, tmp_hrec->radius);
    }
    else {
        computePlaneVtx(plane_vtxs, query_ray_hrec.p, query_ray_hrec.normal, 1);
    }
    vb_plane_.update(plane_vtxs.size(), plane_vtxs.data());

    // Update plane normal
    if (tmp_hrec) {
        std::array<aten::vec4, 2> plane_nml_vtxs = {
            aten::vec4(tmp_hrec->center, 1),
            aten::vec4(tmp_hrec->center, 1) + aten::vec4(tmp_hrec->normal, 0),
        };
        vb_plane_nml_.update(plane_nml_vtxs.size(), plane_nml_vtxs.data());
    }
    else {
        std::array<aten::vec4, 2> plane_nml_vtxs = {
            aten::vec4(query_ray_hrec.p, 1),
            aten::vec4(query_ray_hrec.p, 1) + aten::vec4(query_ray_hrec.normal, 0),
        };
        vb_plane_nml_.update(plane_nml_vtxs.size(), plane_nml_vtxs.data());
    }

    // Update Query ray
    std::array<aten::vec4, 2> query_ray_vtxs = {
        aten::vec4(query_ray.org, 1),
        aten::vec4(query_ray_hrec.p, 1),
    };
    vb_query_ray_.update(query_ray_vtxs.size(), query_ray_vtxs.data());

    // Update Sample ray
    std::array<aten::vec4, 2> sample_ray_vtxs = {
        aten::vec4(sample_ray.org, 1),
        aten::vec4(sample_ray_hit_pos, 1),
    };
    vb_sample_ray_.update(sample_ray_vtxs.size(), sample_ray_vtxs.data());
}

void FeatureLineSampleRay::draw(
    uint32_t count,
    const aten::camera* cam)
{
    auto camparam = cam->param();

    // TODO
    camparam.znear = real(0.1);
    camparam.zfar = real(10000.0);

    aten::mat4 mtxW2V;
    aten::mat4 mtxV2C;

    aten::mat4 mtxL2W;
    mtxL2W.asScale(SceneScaleFactor);

    mtxW2V.lookat(
        camparam.origin,
        camparam.center,
        camparam.up);

    mtxV2C.perspective(
        camparam.znear,
        camparam.zfar,
        camparam.vfov,
        camparam.aspect);

    aten::mat4 mtxW2C = mtxV2C * mtxW2V;

    auto hMtxW2C = shader_.getHandle("mtxW2C");
    CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

    auto hMtxL2W = shader_.getHandle("mtxL2W");
    auto hColor = shader_.getHandle("color");

    CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, mtxL2W.data()));

    switch (count % 4) {
    case 0:
        CALL_GL_API(::glUniform3f(hColor, 1.0f, 1.0f, 1.0f));
        break;
    case 1:
        CALL_GL_API(::glUniform3f(hColor, 0.5f, 0.5f, 0.5f));
        break;
    case 2:
        CALL_GL_API(::glUniform3f(hColor, 0.5f, 0.5f, 0.0f));
        break;
    case 3:
        CALL_GL_API(::glUniform3f(hColor, 0.0f, 0.5f, 0.5f));
        break;
    }
    ib_plane_.draw(vb_plane_, aten::Primitive::Triangles, 0, 2);

    CALL_GL_API(::glUniform3f(hColor, 1.0f, 0.0f, 0.0f));
    ib_line_.draw(vb_query_ray_, aten::Primitive::Lines, 0, 1);

    CALL_GL_API(::glUniform3f(hColor, 0.0f, 1.0f, 0.0f));
    ib_line_.draw(vb_sample_ray_, aten::Primitive::Lines, 0, 1);

    CALL_GL_API(::glUniform3f(hColor, 0.0f, 0.0f, 0.0f));
    ib_line_.draw(vb_plane_nml_, aten::Primitive::Lines, 0, 1);
}
