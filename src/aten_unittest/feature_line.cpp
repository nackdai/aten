#include <gtest/gtest.h>

#include "aten.h"
#include "renderer/npr/feature_line.h"

TEST(feature_line_test, GenerateDisc)
{
    constexpr float line_width = float(2);
    constexpr float pixel_width = float(3);

    const aten::ray query_ray(
        aten::vec3(0, 0, 0),
        aten::vec3(0, 1, 0));

    // Plane at distance 1 from camera.
    // And this plane has the opposite normal with query ray.
    const aten::vec4 plane(0, -1, 0, 1);

    const auto res_hit_pos = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(plane, query_ray);
    const auto expected_disc_center = std::get<1>(res_hit_pos);
    const auto expected_disc_normal = query_ray.dir;
    const auto expected_disc_radius = line_width * pixel_width;

    const auto disc = AT_NAME::npr::FeatureLine::GenerateDisc(query_ray, line_width, pixel_width);

    EXPECT_FLOAT_EQ(disc.center.x, expected_disc_center.x);
    EXPECT_FLOAT_EQ(disc.center.y, expected_disc_center.y);
    EXPECT_FLOAT_EQ(disc.center.z, expected_disc_center.z);
    EXPECT_FLOAT_EQ(disc.normal.x, expected_disc_normal.x);
    EXPECT_FLOAT_EQ(disc.normal.y, expected_disc_normal.y);
    EXPECT_FLOAT_EQ(disc.normal.z, expected_disc_normal.z);
    EXPECT_FLOAT_EQ(disc.radius, expected_disc_radius);
}

TEST(feature_line_test, ComputeDiscAtQueryRayHitPoint)
{
    aten::vec3 query_ray_hit_pos(1, 2, 3);
    aten::vec3 ray_dir(0, 1, 0);
    constexpr auto previous_disc_radius = 1.0f;
    constexpr auto current_hit_distance = 1.5f;
    constexpr auto accumulatedDistanceFromCameraWithoutCurrentHitDistance = 10.0f;

    const auto disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
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

    EXPECT_FLOAT_EQ(disc.accumulated_distance, accumulatedDistanceFromCameraWithoutCurrentHitDistance);

    constexpr auto expected_radius = (current_hit_distance + accumulatedDistanceFromCameraWithoutCurrentHitDistance) / accumulatedDistanceFromCameraWithoutCurrentHitDistance;
    EXPECT_NEAR(disc.radius, expected_radius, 1e-7);
}

TEST(feature_line_test, StoreRayToDescAndGet)
{
    aten::ray ray(
        aten::vec3(1, 2, 3),
        aten::vec3(4, 5, 6));

    AT_NAME::npr::FeatureLine::SampleRayDesc desc;
    AT_NAME::npr::FeatureLine::StoreRayInSampleRayDesc(desc, ray);

    EXPECT_FLOAT_EQ(desc.ray_org_x, ray.org.x);
    EXPECT_FLOAT_EQ(desc.ray_org_y, ray.org.y);
    EXPECT_FLOAT_EQ(desc.ray_org_z, ray.org.z);

    EXPECT_FLOAT_EQ(desc.ray_dir.x, ray.dir.x);
    EXPECT_FLOAT_EQ(desc.ray_dir.y, ray.dir.y);
    EXPECT_FLOAT_EQ(desc.ray_dir.z, ray.dir.z);

    const auto stored_ray = AT_NAME::npr::FeatureLine::ExtractRayFromSampleRayDesc(desc);

    EXPECT_FLOAT_EQ(stored_ray.org.x, ray.org.x);
    EXPECT_FLOAT_EQ(stored_ray.org.y, ray.org.y);
    EXPECT_FLOAT_EQ(stored_ray.org.z, ray.org.z);
    EXPECT_FLOAT_EQ(stored_ray.dir.x, ray.dir.x);
    EXPECT_FLOAT_EQ(stored_ray.dir.y, ray.dir.y);
    EXPECT_FLOAT_EQ(stored_ray.dir.z, ray.dir.z);
}

TEST(feature_line_test, ComputePlane)
{
    aten::hitrecord hrec;
    hrec.normal = aten::vec3(0, 1, 0);
    hrec.p = aten::vec3(0, 1, 0);

    const auto d = -dot(hrec.normal, hrec.p);

    const auto plane = AT_NAME::npr::FeatureLine::ComputePlane(hrec);

    EXPECT_FLOAT_EQ(plane.x, hrec.normal.x);
    EXPECT_FLOAT_EQ(plane.y, hrec.normal.y);
    EXPECT_FLOAT_EQ(plane.z, hrec.normal.z);
    EXPECT_FLOAT_EQ(plane.w, d);
}

TEST(feature_line_test, ComputeRayHitPositionOnPlane)
{
    aten::ray ray;
    aten::vec4 plane(0, 1, 0, 0);

    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(1, -1, 0));
    auto result = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(plane, ray);
    auto is_hit = std::get<0>(result);
    auto& hit_pos = std::get<1>(result);

    EXPECT_TRUE(is_hit);
    EXPECT_NEAR(hit_pos.x, 1, 1e-7);
    EXPECT_NEAR(hit_pos.y, 0, 1e-7);
    EXPECT_FLOAT_EQ(hit_pos.z, 0);

    ray.org = aten::vec3(0, -1, 0);
    ray.dir = normalize(aten::vec3(1, 1, 0));
    result = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    hit_pos = std::get<1>(result);

    EXPECT_TRUE(is_hit);
    EXPECT_NEAR(hit_pos.x, 1, 1e-7);
    EXPECT_NEAR(hit_pos.y, 0, 1e-7);
    EXPECT_FLOAT_EQ(hit_pos.z, 0);

    // Ray is parallel to plane.
    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(1, 0, 0));
    result = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    EXPECT_FALSE(is_hit);

    // Hit point is opossite direction from ray.
    ray.org = aten::vec3(0, 1, 0);
    ray.dir = normalize(aten::vec3(0, 1, 0));
    result = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(plane, ray);
    is_hit = std::get<0>(result);
    EXPECT_FALSE(is_hit);
}

TEST(feature_line_test, GenerateSampleRay)
{
    constexpr auto sampler_r0 = float(0.5);
    constexpr auto sampler_r1 = float(0.6);

    class TestSampler : public aten::sampler {
    public:
        TestSampler() = default;
        virtual ~TestSampler() = default;
        virtual AT_HOST_DEVICE_API float nextSample() override final { return float(0); }

        virtual AT_HOST_DEVICE_API aten::vec2 nextSample2D() override final
        {
            return aten::vec2(sampler_r0, sampler_r1);
        }
    };
    TestSampler sampler;

    const aten::ray query_ray(aten::vec3(0), aten::vec3(0, 0, -1));

    AT_NAME::npr::FeatureLine::Disc disc;
    disc.center = aten::vec3(0, 1, -1);
    disc.normal = aten::vec3(0, 0, 1);
    disc.radius = float(1);

    auto expected_u = sampler_r0;
    auto expected_v = sampler_r1;

    // [0,1] -> [-1,1]
    expected_u = expected_u * 2 - 1;
    expected_v = expected_v * 2 - 1;

    const auto pos_on_disc = AT_NAME::npr::FeatureLine::ComputeHitPositionOnDisc(
        expected_u, expected_v,
        disc);

    const aten::ray expected_ray(
        query_ray.org,
        static_cast<aten::vec3>(pos_on_disc) - query_ray.org);

    AT_NAME::npr::FeatureLine::SampleRayDesc sample_ray_desc;
    const auto result = AT_NAME::npr::FeatureLine::GenerateSampleRay(sample_ray_desc, sampler, query_ray, disc);

    EXPECT_FLOAT_EQ(result.org.x, expected_ray.org.x);
    EXPECT_FLOAT_EQ(result.org.y, expected_ray.org.y);
    EXPECT_FLOAT_EQ(result.org.z, expected_ray.org.z);
    EXPECT_FLOAT_EQ(result.dir.x, expected_ray.dir.x);
    EXPECT_FLOAT_EQ(result.dir.y, expected_ray.dir.y);
    EXPECT_FLOAT_EQ(result.dir.z, expected_ray.dir.z);

    EXPECT_FLOAT_EQ(expected_u, sample_ray_desc.u);
    EXPECT_FLOAT_EQ(expected_v, sample_ray_desc.v);
}

TEST(feature_line_test, ComputeNextSampleRay)
{
    AT_NAME::npr::FeatureLine::Disc disc;
    disc.radius = float(1);
    disc.center = aten::vec3(0, 0, 1);
    disc.normal = aten::vec3(0, 0, -1);

    AT_NAME::npr::FeatureLine::Disc next_disc;
    next_disc.radius = float(2);
    next_disc.center = aten::vec3(0, 0, 2);
    next_disc.normal = aten::vec3(0, 0, 1);

    AT_NAME::npr::FeatureLine::SampleRayDesc sample_ray_desc;
    sample_ray_desc.u = 0.5;
    sample_ray_desc.v = 0.5;
    sample_ray_desc.prev_ray_hit_pos = aten::vec3(0);
    sample_ray_desc.prev_ray_hit_nml = aten::vec3(0, 1, 0);

    const auto disc_face_ratio = dot(disc.normal, next_disc.normal);
    const auto u = disc_face_ratio >= 0 ? sample_ray_desc.u : -sample_ray_desc.u;
    const auto v = sample_ray_desc.v;
    const auto pos_on_next_disc = AT_NAME::npr::FeatureLine::ComputeHitPositionOnDisc(u, v, next_disc);
    const auto ray_dir = static_cast<aten::vec3>(pos_on_next_disc) - sample_ray_desc.prev_ray_hit_pos;
    const aten::ray expected_sample_ray(
        sample_ray_desc.prev_ray_hit_pos,
        ray_dir,
        sample_ray_desc.prev_ray_hit_nml);

    const auto result = AT_NAME::npr::FeatureLine::ComputeNextSampleRay(
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

TEST(feature_line_test, ComputeHitPositionOnDisc)
{
    AT_NAME::npr::FeatureLine::Disc disc;
    disc.radius = float(1);
    disc.center = aten::vec3(1, 2, 3);
    disc.normal = aten::vec3(0, 1, 0);

    constexpr float u = float(0.5);
    constexpr float v = float(0.5);

    aten::vec4 expected_pos(u, v, 0, 1);
    expected_pos *= disc.radius;
    aten::vec3 n = disc.normal;

    aten::vec3 t, b;
    aten::tie(t, b) = aten::GetTangentCoordinate(n);

    aten::mat4 mtx_axes(t, b, n);
    expected_pos = mtx_axes.applyXYZ(expected_pos);
    expected_pos = static_cast<aten::vec3>(expected_pos) + disc.center;

    const auto result = AT_NAME::npr::FeatureLine::ComputeHitPositionOnDisc(u, v, disc);

    EXPECT_FLOAT_EQ(expected_pos.x, result.x);
    EXPECT_FLOAT_EQ(expected_pos.y, result.y);
    EXPECT_FLOAT_EQ(expected_pos.z, result.z);
}

TEST(feature_line_test, ProjectPointOnRay)
{
    const aten::ray ray(
        aten::vec3(0, 0, 0),
        aten::vec3(1, 0, 0));
    const aten::vec3 point(1, 1, 0);

    const auto result = AT_NAME::npr::FeatureLine::ProjectPointOnRay(point, ray, nullptr);

    EXPECT_FLOAT_EQ(result, 1);
}

TEST(feature_line_test, ComputeDistanceBetweenProjectedPositionOnRayAndRayOrigin)
{
    const aten::ray ray(
        aten::vec3(0, 0, 0),
        aten::vec3(1, 0, 0));
    const aten::vec3 point(1, 1, 0);

    aten::vec3 pos_on_ray;
    (void)AT_NAME::npr::FeatureLine::ProjectPointOnRay(point, ray, &pos_on_ray);
    const auto expected_distance = length(pos_on_ray - ray.org);

    const auto result = AT_NAME::npr::FeatureLine::ComputeDistanceBetweenProjectedPositionOnRayAndRayOrigin(point, ray);

    EXPECT_FLOAT_EQ(result, expected_distance);
}


TEST(feature_line_test, ComputeDepthThreshold)
{
    const aten::vec3 p(0, 0, 0);

    aten::hitrecord hrec_q;
    {
        hrec_q.p = aten::vec3(10, 10, 0);
        hrec_q.normal = normalize(aten::vec3(0, -1, 0));
    }
    const float depth_q = 10;

    aten::hitrecord hrec_s;
    {
        hrec_s.p = aten::vec3(-1, 1, 0);
        hrec_s.normal = normalize(aten::vec3(1, 0, 0));
    }
    const float depth_s = 1;

    const float scale_factor = float(2);

    const auto p_q = hrec_q.p - p;
    const auto p_s = hrec_s.p - p;
    const auto& n_closest = hrec_s.normal;
    const auto max_depth = std::max(depth_q, depth_s);
    const auto div = AT_NAME::abs(dot(p_q, n_closest));
    const auto expected_threshold = scale_factor * max_depth * length(p_s - p_q) / div;

    const auto threshold = AT_NAME::npr::FeatureLine::ComputeDepthThreshold(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);

    EXPECT_FLOAT_EQ(threshold, expected_threshold);

    hrec_s.normal = normalize(aten::vec3(0, 0, 1));
    const auto threshold_flt_max = AT_NAME::npr::FeatureLine::ComputeDepthThreshold(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_FLOAT_EQ(threshold_flt_max, FLT_MAX);
}

TEST(feature_line_test, EvaluateFeatureLineMetrics)
{
    // Mesh Id.
    auto is_mesh = AT_NAME::npr::FeatureLine::EvaluateMeshIdMetric(0, 1);
    EXPECT_TRUE(is_mesh);
    is_mesh = AT_NAME::npr::FeatureLine::EvaluateMeshIdMetric(0, 0);
    EXPECT_FALSE(is_mesh);

    // Albedo.
    auto is_albedo = AT_NAME::npr::FeatureLine::EvaluateAlbedoMetric(float(0.01), aten::vec4(1, 0, 0, 0), aten::vec4(0, 1, 0, 0));
    EXPECT_TRUE(is_albedo);
    is_albedo = AT_NAME::npr::FeatureLine::EvaluateAlbedoMetric(float(0.01), aten::vec4(1, 0, 0, 0), aten::vec4(1, 0, 0, 0));
    EXPECT_FALSE(is_albedo);

    // Normal.
    auto is_normal = AT_NAME::npr::FeatureLine::EvaluateNormalMetric(
        float(0.01),
        normalize(aten::vec3(1, 1, 0)), normalize(aten::vec3(1, 0, 0)));
    EXPECT_TRUE(is_normal);
    is_normal = AT_NAME::npr::FeatureLine::EvaluateNormalMetric(
        float(0.01),
        normalize(aten::vec3(1, 0, 0)), normalize(aten::vec3(1, 0, 0)));
    EXPECT_FALSE(is_normal);

    // Depth.
    const aten::vec3 p(0, 0, 0);

    aten::hitrecord hrec_q;
    {
        hrec_q.p = aten::vec3(10, 10, 0);
        hrec_q.normal = normalize(aten::vec3(0, -1, 0));
    }
    const float depth_q = 3;

    aten::hitrecord hrec_s;
    {
        hrec_s.p = aten::vec3(-1, 1, 0);
        hrec_s.normal = normalize(aten::vec3(1, 0, 0));
    }
    const float depth_s = 1;

    const float scale_factor = float(0.01);

    auto is_depth = AT_NAME::npr::FeatureLine::EvaluateDepthMetric(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_TRUE(is_depth);

    hrec_s.normal = normalize(aten::vec3(0, 0, 1));
    is_depth = AT_NAME::npr::FeatureLine::EvaluateDepthMetric(
        p,
        scale_factor,
        hrec_q, hrec_s,
        depth_q, depth_s);
    EXPECT_FALSE(is_depth);
}

TEST(feature_line_test, IsInLineWidth)
{
    const aten::ray ray(
        aten::vec3(0, 0, 0),
        aten::vec3(1, 0, 0));
    const aten::vec3 point(1, 1, 0);

    float screen_line_width = float(100);
    const float accumulatedDistance = float(2);
    const float pixelWidth = float(0.5);

    auto is_in_line_width = AT_NAME::npr::FeatureLine::IsInLineWidth(screen_line_width, ray, point, accumulatedDistance, pixelWidth);
    EXPECT_TRUE(is_in_line_width);

    screen_line_width = float(0.0001);
    is_in_line_width = AT_NAME::npr::FeatureLine::IsInLineWidth(screen_line_width, ray, point, accumulatedDistance, pixelWidth);
    EXPECT_FALSE(is_in_line_width);
}
