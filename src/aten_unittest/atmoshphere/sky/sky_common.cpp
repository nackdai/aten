#include "sky_test_fixture.h"

TEST_F(SkyTest, safe_sqrt)
{
    EXPECT_EQ(aten::sky::safe_sqrt(0.0F), 0.0F);
    EXPECT_EQ(aten::sky::safe_sqrt(-1.0F), 0.0F);
}

TEST_F(SkyTest, DistanceToTopAtmosphereBoundary)
{
    constexpr float r = BottomRadius * 0.2F + TopRadius * 0.8F;

    // Vertical ray, looking top.
    EXPECT_NEAR(
        TopRadius - r,
        aten::sky::DistanceToTopAtmosphereBoundary(atmosphere_, r, 1.0F),
        1.0F);
    // Horizontal ray.
    EXPECT_NEAR(
        aten::sqrt(TopRadius * TopRadius - r * r),
        DistanceToTopAtmosphereBoundary(atmosphere_, r, 0.0F),
        1.0F);
}

TEST_F(SkyTest, DistanceToBottomAtmosphereBoundary)
{
    constexpr float r = BottomRadius * 0.2F + TopRadius * 0.8F;

    // Vertical ray, looking down.
    EXPECT_NEAR(
        r - BottomRadius,
        aten::sky::DistanceToBottomAtmosphereBoundary(atmosphere_, r, -1.0F),
        1.0F);
}

TEST_F(SkyTest, RayIntersectsGround)
{
    constexpr float r = BottomRadius * 0.9F + TopRadius * 0.1F;

    // Vertical ray looking up should not intersect ground
    EXPECT_FALSE(aten::sky::RayIntersectsGround(atmosphere_, r, 1.0F));

    // Ray above horizon should not intersect ground
    float mu_horizon = -aten::sqrt(1.0F - (BottomRadius / r) * (BottomRadius / r));
    EXPECT_FALSE(aten::sky::RayIntersectsGround(atmosphere_, r, mu_horizon + Epsilon));

    // Ray below horizon should intersect ground
    EXPECT_TRUE(aten::sky::RayIntersectsGround(atmosphere_, r, mu_horizon - Epsilon));

    // Vertical ray looking down should intersect ground
    EXPECT_TRUE(aten::sky::RayIntersectsGround(atmosphere_, r, -1.0F));
}

TEST_F(SkyTest, DistanceToNearestAtmosphereBoundary)
{
    constexpr float r = BottomRadius * 0.2F + TopRadius * 0.8F;

    // Vertical ray, looking top.
    EXPECT_NEAR(
        TopRadius - r,
        aten::sky::DistanceToNearestAtmosphereBoundary(
            atmosphere_, r, 1.0F,
            aten::sky::RayIntersectsGround(atmosphere_, r, 1.0F)),
        1.0F);

    // Horizontal ray.
    EXPECT_NEAR(
        aten::sqrt(TopRadius * TopRadius - r * r),
        aten::sky::DistanceToNearestAtmosphereBoundary(
            atmosphere_, r, 0.0F,
            aten::sky::RayIntersectsGround(atmosphere_, r, 0.0F)),
        1.0F);

    // Vertical ray, looking down.
    EXPECT_NEAR(
        r - BottomRadius,
        aten::sky::DistanceToNearestAtmosphereBoundary(
            atmosphere_, r, -1.0F,
            aten::sky::RayIntersectsGround(atmosphere_, r, -1.0F)),
        1.0F);
}

TEST_F(SkyTest, PhaseFunctions)
{
    // Test Rayleigh phase function integration over solid angle
    // The integral of any phase function over all solid angles should be 1
    float rayleigh_integral = 0.0F;
    float mie_integral = 0.0F;
    constexpr int32_t N = 100;
    constexpr float PI = 3.14159265359F;

    for (int32_t i = 0; i < N; ++i) {
        float theta = (i + 0.5F) * PI / N;
        float domega = aten::sin(theta) * (PI / N) * (2.0F * PI);
        float cos_theta = aten::cos(theta);

        rayleigh_integral += aten::sky::RayleighPhaseFunction(cos_theta) * domega;
        mie_integral += aten::sky::MiePhaseFunction(0.8F, cos_theta) * domega;
    }

    // The relative error tolerance is larger due to numerical integration
    EXPECT_NEAR(1.0F, rayleigh_integral, 2.0F * Epsilon);
    EXPECT_NEAR(1.0F, mie_integral, 2.0F * Epsilon);
}
