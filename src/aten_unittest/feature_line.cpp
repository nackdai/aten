#include <gtest/gtest.h>

#include "aten.h"
#include "renderer/feature_line.h"

TEST(feature_line_test, GenerateDiscTest)
{
    constexpr real line_width = real(2);
    constexpr real pixel_width = real(3);

    const aten::ray query_ray(
        aten::vec3(0, 0, 0),
        aten::vec3(0, 1, 0));

    // Plane at distance 1 from camera.
    // And this plane has the opposite normal with query ray.
    const aten::vec4 plane(0, -1, 0, 1);

    const auto res_hit_pos = aten::FeatureLine::computeRayHitPosOnPlane(plane, query_ray);
    const auto expected_disc_center = std::get<1>(res_hit_pos);
    const auto expected_disc_normal = query_ray.dir;
    const auto expected_disc_radius = line_width * pixel_width;

    const auto disc = aten::FeatureLine::generateDisc(query_ray, line_width, pixel_width);

    EXPECT_FLOAT_EQ(disc.center.x, expected_disc_center.x);
    EXPECT_FLOAT_EQ(disc.center.y, expected_disc_center.y);
    EXPECT_FLOAT_EQ(disc.center.z, expected_disc_center.z);
    EXPECT_FLOAT_EQ(disc.normal.x, expected_disc_normal.x);
    EXPECT_FLOAT_EQ(disc.normal.y, expected_disc_normal.y);
    EXPECT_FLOAT_EQ(disc.normal.z, expected_disc_normal.z);
    EXPECT_FLOAT_EQ(disc.radius, expected_disc_radius);
}

TEST(feature_line_test, ComputeNextDiscTest)
{
    aten::vec3 query_ray_hit_pos(1, 2, 3);
    aten::vec3 ray_dir(0, 1, 0);
    constexpr auto previous_disc_radius = 1.0f;
    constexpr auto current_hit_distance = 1.5f;
    constexpr auto accumulatedDistanceFromCameraWithoutCurrentHitDistance = 10.0f;

    const auto disc = aten::FeatureLine::computeNextDisc(
        query_ray_hit_pos,
        ray_dir,
        previous_disc_radius,
        current_hit_distance,
        accumulatedDistanceFromCameraWithoutCurrentHitDistance);

    EXPECT_FLOAT_EQ(disc.center.x, 1);
    EXPECT_FLOAT_EQ(disc.center.y, 2);
    EXPECT_FLOAT_EQ(disc.center.z, 3);
    EXPECT_FLOAT_EQ(disc.normal.x, 0);
    EXPECT_FLOAT_EQ(disc.normal.y, -1);
    EXPECT_FLOAT_EQ(disc.normal.z, 0);

    constexpr auto expected_radius = (current_hit_distance + accumulatedDistanceFromCameraWithoutCurrentHitDistance) / accumulatedDistanceFromCameraWithoutCurrentHitDistance;
    EXPECT_NEAR(disc.radius, expected_radius, 1e-7);
}

TEST(feature_line_test, ComputePlaneTest)
{
    aten::hitrecord hrec;
    hrec.normal = aten::vec3(0, 1, 0);
    hrec.p = aten::vec3(0, 1, 0);

    const auto d = -dot(hrec.normal, hrec.p);

    const auto plane = aten::FeatureLine::computePlane(hrec);

    EXPECT_FLOAT_EQ(plane.x, hrec.normal.x);
    EXPECT_FLOAT_EQ(plane.y, hrec.normal.y);
    EXPECT_FLOAT_EQ(plane.z, hrec.normal.z);
    EXPECT_FLOAT_EQ(plane.w, d);
}

TEST(feature_line_test, ComputeRayHitPosOnPlaneTest)
{
    aten::ray ray;
    aten::vec4 plane(0, 1, 0, 0);

    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(1, -1, 0));
    auto result = aten::FeatureLine::computeRayHitPosOnPlane(plane, ray);
    auto is_hit = std::get<0>(result);
    auto& hit_pos = std::get<1>(result);

    EXPECT_TRUE(is_hit);
    EXPECT_NEAR(hit_pos.x, 1, 1e-7);
    EXPECT_NEAR(hit_pos.y, 0, 1e-7);
    EXPECT_FLOAT_EQ(hit_pos.z, 0);

    ray.org = aten::vec3(0, -1, 0);
    ray.dir = normalize(aten::vec3(1, 1, 0));
    result = aten::FeatureLine::computeRayHitPosOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    hit_pos = std::get<1>(result);

    EXPECT_TRUE(is_hit);
    EXPECT_NEAR(hit_pos.x, 1, 1e-7);
    EXPECT_NEAR(hit_pos.y, 0, 1e-7);
    EXPECT_FLOAT_EQ(hit_pos.z, 0);

    // Ray is parallel to plane.
    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(1, 0, 0));
    result = aten::FeatureLine::computeRayHitPosOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    EXPECT_FALSE(is_hit);

    // Hit point is opossite direction from ray.
    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(0, 1, 0));
    result = aten::FeatureLine::computeRayHitPosOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    EXPECT_FALSE(is_hit);
}

TEST(feature_line_test, GenerateSampleRayTest)
{
    constexpr auto sampler_r0 = real(0.5);
    constexpr auto sampler_r1 = real(0.6);

    class TestSampler : public aten::sampler {
    public:
        TestSampler() = default;
        virtual ~TestSampler() = default;
        virtual AT_DEVICE_API real nextSample() override final { return real(0); }

        virtual AT_DEVICE_API aten::vec2 nextSample2D() override final
        {
            return aten::vec2(sampler_r0, sampler_r1);
        }
    };
    TestSampler sampler;

    const aten::ray query_ray(aten::vec3(0), aten::vec3(0, 0, -1));

    aten::FeatureLine::Disc disc;
    disc.center = aten::vec3(0, 1, -1);
    disc.normal = aten::vec3(0, 0, 1);
    disc.radius = real(1);

    auto expected_u = sampler_r0;
    auto expected_v = sampler_r1;

    // [0,1] -> [-1,1]
    expected_u = expected_u * 2 - 1;
    expected_v = expected_v * 2 - 1;

    const auto pos_on_disc = aten::FeatureLine::computePosOnDisc(
        expected_u, expected_v,
        disc);

    const aten::ray expected_ray(
        query_ray.org,
        static_cast<aten::vec3>(pos_on_disc) - query_ray.org);

    aten::FeatureLine::SampleRayDesc sample_ray_desc;
    const auto result = aten::FeatureLine::generateSampleRay(sample_ray_desc, sampler, query_ray, disc);

    EXPECT_FLOAT_EQ(result.org.x, expected_ray.org.x);
    EXPECT_FLOAT_EQ(result.org.y, expected_ray.org.y);
    EXPECT_FLOAT_EQ(result.org.z, expected_ray.org.z);
    EXPECT_FLOAT_EQ(result.dir.x, expected_ray.dir.x);
    EXPECT_FLOAT_EQ(result.dir.y, expected_ray.dir.y);
    EXPECT_FLOAT_EQ(result.dir.z, expected_ray.dir.z);

    EXPECT_FLOAT_EQ(expected_u, sample_ray_desc.u);
    EXPECT_FLOAT_EQ(expected_v, sample_ray_desc.v);
}

TEST(feature_line_test, ComputeNextSampleRayTest)
{
    aten::FeatureLine::Disc disc;
    disc.radius = real(1);
    disc.center = aten::vec3(0, 0, 1);
    disc.normal = aten::vec3(0, 0, -1);

    aten::FeatureLine::Disc next_disc;
    next_disc.radius = real(2);
    next_disc.center = aten::vec3(0, 0, 2);
    next_disc.normal = aten::vec3(0, 0, 1);

    aten::FeatureLine::SampleRayDesc sample_ray_desc;
    sample_ray_desc.u = 0.5;
    sample_ray_desc.v = 0.5;
    sample_ray_desc.prev_ray_hit_pos = aten::vec3(0);
    sample_ray_desc.prev_ray_hit_nml = aten::vec3(0, 1, 0);

    const auto disc_face_ratio = dot(disc.normal, next_disc.normal);
    const auto u = disc_face_ratio >= 0 ? sample_ray_desc.u : -sample_ray_desc.u;
    const auto v = sample_ray_desc.v;
    const auto pos_on_next_disc = aten::FeatureLine::computePosOnDisc(u, v, next_disc);
    const auto ray_dir = static_cast<aten::vec3>(pos_on_next_disc) - sample_ray_desc.prev_ray_hit_pos;
    const aten::ray expected_sample_ray(
        sample_ray_desc.prev_ray_hit_pos,
        ray_dir,
        sample_ray_desc.prev_ray_hit_nml);

    const auto result = aten::FeatureLine::computeNextSampleRay(
        sample_ray_desc,
        disc,
        next_disc);

    const auto is_hit = std::get<0>(result);
    const auto& sample_ray = std::get<1>(result);

    EXPECT_TRUE(is_hit);
    EXPECT_FLOAT_EQ(sample_ray.dir.x, expected_sample_ray.dir.x);
    EXPECT_FLOAT_EQ(sample_ray.dir.y, expected_sample_ray.dir.y);
    EXPECT_FLOAT_EQ(sample_ray.dir.z, expected_sample_ray.dir.z);
    EXPECT_FLOAT_EQ(sample_ray.org.x, expected_sample_ray.org.x);
    EXPECT_FLOAT_EQ(sample_ray.org.y, expected_sample_ray.org.y);
    EXPECT_FLOAT_EQ(sample_ray.org.z, expected_sample_ray.org.z);
}

TEST(feature_line_test, ComputePosOnDiscTest)
{
    aten::FeatureLine::Disc disc;
    disc.radius = real(1);
    disc.center = aten::vec3(1, 2, 3);
    disc.normal = aten::vec3(0, 1, 0);

    constexpr real u = real(0.5);
    constexpr real v = real(0.5);

    aten::vec4 expected_pos(u, v, 0, 1);
    expected_pos *= disc.radius;
    aten::vec3 n = disc.normal;
    const auto t = aten::getOrthoVector(n);
    const auto b = cross(n, t);
    aten::mat4 mtx_axes(t, b, n);
    expected_pos = mtx_axes.applyXYZ(expected_pos);
    expected_pos = static_cast<aten::vec3>(expected_pos) + disc.center;

    const auto result = aten::FeatureLine::computePosOnDisc(u, v, disc);

    EXPECT_FLOAT_EQ(expected_pos.x, result.x);
    EXPECT_FLOAT_EQ(expected_pos.y, result.y);
    EXPECT_FLOAT_EQ(expected_pos.z, result.z);
}

TEST(feature_line_test, ComputeDistanceBetweenPointAndRayTest)
{
    const aten::ray ray(
        aten::vec3(0, 0, 0),
        aten::vec3(1, 0, 0));
    const aten::vec3 point(1, 1, 0);

    const auto result = aten::FeatureLine::computeDistanceBetweenPointAndRay(point, ray, nullptr);

    EXPECT_FLOAT_EQ(result, 1);
}

TEST(feature_line_test, ComputeThresholdDepthTest)
{
    const aten::vec3 p(0, 0, 0);

    aten::hitrecord hrec_q;
    {
        hrec_q.p = aten::vec3(10, 10, 0);
        hrec_q.normal = normalize(aten::vec3(0, -1, 0));
    }
    const real depth_q = 10;

    aten::hitrecord hrec_s;
    {
        hrec_s.p = aten::vec3(-1, 1, 0);
        hrec_s.normal = normalize(aten::vec3(1, 0, 0));
    }
    const real depth_s = 1;

    const real scale_factor = real(2);

    const auto p_q = hrec_q.p - p;
    const auto p_s = hrec_s.p - p;
    const auto& n_closest = hrec_s.normal;
    const auto max_depth = std::max(depth_q, depth_s);
    const auto div = AT_NAME::abs(dot(p_q, n_closest));
    const auto expected_threshold = scale_factor * max_depth * length(p_s - p_q) / div;

    const auto threshold = aten::FeatureLine::computeThresholdDepth(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);

    EXPECT_FLOAT_EQ(threshold, expected_threshold);

    hrec_s.normal = normalize(aten::vec3(0, 0, 1));
    const auto threshold_flt_max = aten::FeatureLine::computeThresholdDepth(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_FLOAT_EQ(threshold_flt_max, FLT_MAX);
}

TEST(feature_line_test, EvaluateFeatureLineMetricsTest)
{
    // Mesh Id.
    auto is_mesh = aten::FeatureLine::evaluateMeshIdMetric(0, 1);
    EXPECT_TRUE(is_mesh);
    is_mesh = aten::FeatureLine::evaluateMeshIdMetric(0, 0);
    EXPECT_FALSE(is_mesh);

    // Albedo.
    auto is_albedo = aten::FeatureLine::evaluateAlbedoMetric(real(0.01), aten::vec4(1, 0, 0, 0), aten::vec4(0, 1, 0, 0));
    EXPECT_TRUE(is_albedo);
    is_albedo = aten::FeatureLine::evaluateAlbedoMetric(real(0.01), aten::vec4(1, 0, 0, 0), aten::vec4(1, 0, 0, 0));
    EXPECT_FALSE(is_albedo);

    // Normal.
    auto is_normal = aten::FeatureLine::evaluateNormalMetric(
        real(0.01),
        normalize(aten::vec3(1, 1, 0)), normalize(aten::vec3(1, 0, 0)));
    EXPECT_TRUE(is_normal);
    is_normal = aten::FeatureLine::evaluateNormalMetric(
        real(0.01),
        normalize(aten::vec3(1, 0, 0)), normalize(aten::vec3(1, 0, 0)));
    EXPECT_FALSE(is_normal);

    // Depth.
    const aten::vec3 p(0, 0, 0);

    aten::hitrecord hrec_q;
    {
        hrec_q.p = aten::vec3(10, 10, 0);
        hrec_q.normal = normalize(aten::vec3(0, -1, 0));
    }
    const real depth_q = 3;

    aten::hitrecord hrec_s;
    {
        hrec_s.p = aten::vec3(-1, 1, 0);
        hrec_s.normal = normalize(aten::vec3(1, 0, 0));
    }
    const real depth_s = 1;

    const real scale_factor = real(0.01);

    auto is_depth = aten::FeatureLine::evaluateDepthMetric(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_TRUE(is_depth);

    hrec_s.normal = normalize(aten::vec3(0, 0, 1));
    is_depth = aten::FeatureLine::evaluateDepthMetric(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_FALSE(is_depth);
}

TEST(feature_line_test, IsInLineWidthTest)
{
    const aten::ray ray(
        aten::vec3(0, 0, 0),
        aten::vec3(1, 0, 0));
    const aten::vec3 point(1, 1, 0);

    real screen_line_width = real(100);
    const real accumulatedDistance = real(2);
    const real pixelWidth = real(0.5);

    auto is_in_line_width = aten::FeatureLine::isInLineWidth(screen_line_width, ray, point, accumulatedDistance, pixelWidth);
    EXPECT_TRUE(is_in_line_width);

    screen_line_width = real(0.0001);
    is_in_line_width = aten::FeatureLine::isInLineWidth(screen_line_width, ray, point, accumulatedDistance, pixelWidth);
    EXPECT_FALSE(is_in_line_width);
}
