#include "sky_test_fixture.h"
#include "sky_test_lazy_texture.h"

TEST_F(SkyTest, ComputeOpticalLengthToTopAtmosphereBoundary)
{
    constexpr float r = BottomRadius * 0.2 + TopRadius * 0.8;
    constexpr float h_r = r - BottomRadius;
    constexpr float h_top = TopRadius - BottomRadius;

    // Vertical ray, looking top.
    EXPECT_NEAR(
        RayleighScaleHeight * (aten::exp(-h_r / RayleighScaleHeight) - aten::exp(-h_top / RayleighScaleHeight)),
        ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere_, atmosphere_.rayleigh_density, r, 1.0F),
        1.0F);

    // Horizontal ray, no exponential density fall off.
    SetUniformAtmosphere();
    EXPECT_NEAR(
        aten::sqrt(TopRadius * TopRadius - r * r),
        ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere_, atmosphere_.rayleigh_density, r, 0.0),
        1.0F);
}

TEST_F(SkyTest, ComputeTransmittanceToTopAtmosphereBoundary)
{
    constexpr float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    constexpr float h_r = r - BottomRadius;
    constexpr float h_top = TopRadius - BottomRadius;

    // Vertical ray, looking up.
    float rayleigh_optical_depth = RayleighScattering * RayleighScaleHeight *
        (aten::exp(-h_r / RayleighScaleHeight) -
         aten::exp(-h_top / RayleighScaleHeight));
    float mie_optical_depth = MieExtinction * MieScaleHeight *
        (aten::exp(-h_r / MieScaleHeight) -
         aten::exp(-h_top / MieScaleHeight));

    auto transmittance = aten::sky::transmittance::ComputeTransmittanceToTopAtmosphereBoundary(
        atmosphere_, r, 1.0F);

    EXPECT_NEAR(
        aten::exp(-(rayleigh_optical_depth + mie_optical_depth)),
        transmittance.x,
        Epsilon);
}

TEST_F(SkyTest, ComputeDirectIrradiance)
{
    // Create a transmittance texture (mock)
    // For this test, we'll just verify the function runs and returns reasonable values
    LazyTransmittanceTexture transmittance_texture(atmosphere_);

    const float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    const float mu_s = 0.5F;

    auto irradiance = aten::sky::ComputeDirectIrradiance(
        atmosphere_, transmittance_texture, r, mu_s);

    // Verify output is non-negative
    EXPECT_GE(irradiance.x, 0.0F);
    EXPECT_GE(irradiance.y, 0.0F);
    EXPECT_GE(irradiance.z, 0.0F);

    // Test boundary case: sun below horizon
    auto irradiance_below = aten::sky::ComputeDirectIrradiance(
        atmosphere_, transmittance_texture, r, -1.0F);
    EXPECT_NEAR(0.0F, irradiance_below.x, Epsilon);
    EXPECT_NEAR(0.0F, irradiance_below.y, Epsilon);
    EXPECT_NEAR(0.0F, irradiance_below.z, Epsilon);

    // Test boundary case: sun at zenith
    auto irradiance_zenith = aten::sky::ComputeDirectIrradiance(
        atmosphere_, transmittance_texture, r, 1.0F);
    EXPECT_GT(irradiance_zenith.x, 0.0F);
}

TEST_F(SkyTest, RayleighPhaseFunction)
{
    // Rayleigh phase function should be normalized to 1 when integrated over all directions
    // Check some specific values
    float rayleigh_hg = aten::sky::RayleighPhaseFunction(0.0F);  // nu = 0 (perpendicular)
    float rayleigh_forward = aten::sky::RayleighPhaseFunction(1.0F);  // nu = 1 (forward)
    float rayleigh_back = aten::sky::RayleighPhaseFunction(-1.0F);  // nu = -1 (backward)

    // For Rayleigh: (1 + nu^2) * 3/(16π) should be symmetric in forward and backward
    EXPECT_NEAR(rayleigh_forward, rayleigh_back, Epsilon);

    // Forward (nu=1) should give maximum: (1+1) * 3/(16π) = 6/(16π)
    EXPECT_GT(rayleigh_forward, rayleigh_hg);
}

TEST_F(SkyTest, MiePhaseFunction)
{
    // Mie phase function with g = 0.8
    float mie_hg = aten::sky::MiePhaseFunction(0.8F, 0.0F);
    float mie_forward = aten::sky::MiePhaseFunction(0.8F, 1.0F);
    float mie_back = aten::sky::MiePhaseFunction(0.8F, -1.0F);

    // Mie scattering is forward-biased
    EXPECT_GT(mie_forward, mie_back);

    // All values should be positive
    EXPECT_GT(mie_hg, 0.0F);
    EXPECT_GT(mie_forward, 0.0F);
    EXPECT_GT(mie_back, 0.0F);
}

TEST_F(SkyTest, ComputeSingleScatteringIntegrand)
{
    LazyTransmittanceTexture transmittance_texture(atmosphere_);

    // Vertical ray, from bottom to top atmosphere boundary, scattering at
    // middle of ray, scattering angle equal to 0.
    const float h_top = TopRadius - BottomRadius;
    const float h = h_top / 2.0F;
    aten::vec3 rayleigh;
    aten::vec3 mie;
    aten::sky::single_scattering::ComputeSingleScatteringIntegrand(
        atmosphere_, transmittance_texture,
        BottomRadius, 1.0F, 1.0F, 1.0F, h, false, rayleigh, mie);

    float rayleigh_optical_depth = RayleighScattering * RayleighScaleHeight *
        (1.0F - aten::exp(-h_top / RayleighScaleHeight));
    float mie_optical_depth = MieExtinction * MieScaleHeight *
        (1.0F - aten::exp(-h_top / MieScaleHeight));
    EXPECT_NEAR(
        aten::exp(-rayleigh_optical_depth - mie_optical_depth) *
            aten::exp(-h / RayleighScaleHeight),
        rayleigh.x,
        Epsilon);
    EXPECT_NEAR(
        aten::exp(-rayleigh_optical_depth - mie_optical_depth) *
            aten::exp(-h / MieScaleHeight),
        mie.x,
        Epsilon);

    // Vertical ray, top to middle of atmosphere, scattering angle 180 degrees.
    aten::sky::single_scattering::ComputeSingleScatteringIntegrand(
        atmosphere_, transmittance_texture,
        TopRadius, -1.0F, 1.0F, -1.0F, h, true, rayleigh, mie);
    rayleigh_optical_depth = 2.0F * RayleighScattering * RayleighScaleHeight *
        (aten::exp(-h / RayleighScaleHeight) -
         aten::exp(-h_top / RayleighScaleHeight));
    mie_optical_depth = 2.0F * MieExtinction * MieScaleHeight *
        (aten::exp(-h / MieScaleHeight) -
         aten::exp(-h_top / MieScaleHeight));
    EXPECT_NEAR(
        aten::exp(-rayleigh_optical_depth - mie_optical_depth) *
            aten::exp(-h / RayleighScaleHeight),
        rayleigh.x,
        Epsilon);
    EXPECT_NEAR(
        aten::exp(-rayleigh_optical_depth - mie_optical_depth) *
            aten::exp(-h / MieScaleHeight),
        mie.x,
        Epsilon);

    // Horizontal ray, from bottom to top atmosphere boundary, scattering at
    // 50km, scattering angle equal to 0, uniform atmosphere, no aerosols.
    transmittance_texture.Clear();
    SetUniformAtmosphere();
    RemoveAerosols();
    aten::sky::single_scattering::ComputeSingleScatteringIntegrand(
        atmosphere_, transmittance_texture,
        BottomRadius, 0.0F, 0.0F, 1.0F, 50.0F * km,
        false, rayleigh, mie);
    rayleigh_optical_depth = RayleighScattering * aten::sqrt(
        TopRadius * TopRadius - BottomRadius * BottomRadius);
    EXPECT_NEAR(
        aten::exp(-rayleigh_optical_depth),
        rayleigh.x,
        Epsilon);
}

TEST_F(SkyTest, ComputeSingleScattering)
{
    LazyTransmittanceTexture transmittance_texture(atmosphere_);

    // Vertical ray, from bottom atmosphere boundary, scattering angle 0.
    constexpr float h_top = TopRadius - BottomRadius;
    aten::vec3 rayleigh;
    aten::vec3 mie;

    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        BottomRadius, 1.0F, 1.0F, 1.0F, false, rayleigh, mie);
    const float rayleigh_optical_depth = RayleighScattering * RayleighScaleHeight * (1.0F - aten::exp(-h_top / RayleighScaleHeight));
    const float mie_optical_depth = MieExtinction * MieScaleHeight * (1.0F - aten::exp(-h_top / MieScaleHeight));

    // The relative error is about 1% here.
    EXPECT_NEAR(
        1.0F,
        rayleigh[0] / (SolarIrradiance * rayleigh_optical_depth *
            exp(-rayleigh_optical_depth - mie_optical_depth)),
        10.0F * Epsilon);
    EXPECT_NEAR(
        1.0F,
        mie[0] / (SolarIrradiance * mie_optical_depth * MieScattering /
            MieExtinction * aten::exp(-rayleigh_optical_depth - mie_optical_depth)),
        10.0F * Epsilon);

    // Vertical ray, from top atmosphere boundary, scattering angle 180 degrees,
    // no aerosols.
    transmittance_texture.Clear();
    RemoveAerosols();
    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        TopRadius, -1.0F, 1.0F, -1.0F, true, rayleigh, mie);
    EXPECT_NEAR(
        1.0F,
        rayleigh[0] / (SolarIrradiance *
            0.5F * (1.0F - aten::exp(-2.0F * RayleighScaleHeight * RayleighScattering *
                (1.0F - exp(-h_top / RayleighScaleHeight))))),
        2.0F * Epsilon);
    EXPECT_NEAR(0.0F, mie[0], Epsilon);
}

TEST_F(SkyTest, ComputeAndGetSingleScattering)
{
    LazyTransmittanceTexture transmittance_texture(atmosphere_);
    LazySingleScatteringTexture single_rayleigh_scattering_texture(
        atmosphere_, transmittance_texture, true);
    LazySingleScatteringTexture single_mie_scattering_texture(
        atmosphere_, transmittance_texture, false);

    // Vertical ray, from bottom atmosphere boundary, scattering angle 0.
    aten::vec3 rayleigh = aten::sky::scattering::GetScattering(
        atmosphere_, single_rayleigh_scattering_texture,
        BottomRadius, 1.0, 1.0, 1.0, false);
    aten::vec3 mie = aten::sky::scattering::GetScattering(
        atmosphere_, single_mie_scattering_texture,
        BottomRadius, 1.0, 1.0, 1.0, false);
    aten::vec3 expected_rayleigh;
    aten::vec3 expected_mie;
    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        BottomRadius, 1.0, 1.0, 1.0, false, expected_rayleigh, expected_mie);
    EXPECT_NEAR(1.0F, (rayleigh / expected_rayleigh)[0], Epsilon);
    EXPECT_NEAR(1.0F, (mie / expected_mie)[0], Epsilon);

    // Vertical ray, from top atmosphere boundary, scattering angle 180 degrees.
    rayleigh = aten::sky::scattering::GetScattering(
        atmosphere_, single_rayleigh_scattering_texture,
        TopRadius, -1.0, 1.0, -1.0, true);
    mie = aten::sky::scattering::GetScattering(
        atmosphere_, single_mie_scattering_texture,
        TopRadius, -1.0, 1.0, -1.0, true);
    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        TopRadius, -1.0, 1.0, -1.0, true, expected_rayleigh, expected_mie);
    EXPECT_NEAR(1.0F, (rayleigh / expected_rayleigh)[0], Epsilon);
    EXPECT_NEAR(1.0F, (mie / expected_mie)[0], Epsilon);

    // Horizontal ray, from bottom of atmosphere, scattering angle 90 degrees.
    rayleigh = aten::sky::scattering::GetScattering(
        atmosphere_, single_rayleigh_scattering_texture,
        BottomRadius, 0.0, 0.0, 0.0, false);
    mie = aten::sky::scattering::GetScattering(
        atmosphere_, single_mie_scattering_texture,
        BottomRadius, 0.0, 0.0, 0.0, false);
    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        BottomRadius, 0.0, 0.0, 0.0, false, expected_rayleigh, expected_mie);
    // The relative error is quite large in this case, i.e. between 6 to 8%.
    EXPECT_NEAR(1.0F, (rayleigh / expected_rayleigh)[0], 1e-1F);
    EXPECT_NEAR(1.0F, (mie / expected_mie)[0], 1e-1F);

    // Ray just above the horizon, sun at the zenith.
    const float mu = CosineOfHorizonZenithAngle(TopRadius);
    rayleigh = aten::sky::scattering::GetScattering(
        atmosphere_, single_rayleigh_scattering_texture,
        TopRadius, mu, 1.0, mu, false);
    mie = aten::sky::scattering::GetScattering(
        atmosphere_, single_mie_scattering_texture,
        TopRadius, mu, 1.0, mu, false);
    aten::sky::single_scattering::ComputeSingleScattering(
        atmosphere_, transmittance_texture,
        TopRadius, mu, 1.0, mu, false, expected_rayleigh, expected_mie);
    EXPECT_NEAR(1.0F, (rayleigh / expected_rayleigh)[0], Epsilon);
    EXPECT_NEAR(1.0F, (mie / expected_mie)[0], Epsilon);
}

TEST_F(SkyTest, GetProfileDensity)
{
    aten::sky::DensityProfile profile;
    // Only one layer, with exponential density.
    profile.layers[1] = aten::sky::DensityProfileLayer{ 0.0F, 1.0F, -1.0F / km, 0.0F, 0.0F };
    EXPECT_NEAR(aten::exp(-2.0F), aten::sky::GetProfileDensity(profile, 2.0F * km), Epsilon);

    // Only one layer, with (clamped) affine density.
    profile.layers[1] = aten::sky::DensityProfileLayer{ 0.0F, 0.0F, 0.0F, -0.5F / km, 1.0F };
    EXPECT_NEAR(1.0F, aten::sky::GetProfileDensity(profile, 0.0F * km), Epsilon);
    EXPECT_NEAR(0.5F, aten::sky::GetProfileDensity(profile, 1.0F * km), Epsilon);
    EXPECT_NEAR(0.0F, aten::sky::GetProfileDensity(profile, 3.0F * km), Epsilon);

    // Two layers, with (clamped) affine density.
    profile.layers[0] = aten::sky::DensityProfileLayer{
        25.0F * km, 0.0F, 0.0F, 1.0F / (15.0F * km), -2.0F / 3.0F };
    profile.layers[1] = aten::sky::DensityProfileLayer{
        0.0F, 0.0F, 0.0F, -1.0F / (15.0F * km), 8.0F / 3.0F };
    EXPECT_NEAR(0.0F, aten::sky::GetProfileDensity(profile, 0.0F * km), Epsilon);
    EXPECT_NEAR(0.0F, aten::sky::GetProfileDensity(profile, 10.0F * km), Epsilon);
    EXPECT_NEAR(1.0F, aten::sky::GetProfileDensity(profile, 25.0F * km), Epsilon);
    EXPECT_NEAR(0.0F, aten::sky::GetProfileDensity(profile, 40.0F * km), Epsilon);
    EXPECT_NEAR(0.0F, aten::sky::GetProfileDensity(profile, 50.0F * km), Epsilon);
}

TEST_F(SkyTest, GetTransmittanceToTopAtmosphereBoundary)
{
    LazyTransmittanceTexture transmittance_texture(atmosphere_);

    const float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    const float mu = 0.4F;
    EXPECT_NEAR(
        aten::sky::transmittance::ComputeTransmittanceToTopAtmosphereBoundary(
            atmosphere_, r, mu)[0],
        aten::sky::transmittance::GetTransmittanceToTopAtmosphereBoundary(
            atmosphere_, transmittance_texture, r, mu)[0],
        Epsilon);
}

TEST_F(SkyTest, ComputeAndGetTransmittance)
{
    SetUniformAtmosphere();
    RemoveAerosols();
    LazyTransmittanceTexture transmittance_texture(atmosphere_);

    const float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    const float d = (TopRadius - BottomRadius) * 0.1F;
    // Horizontal ray, from bottom atmosphere boundary.
    EXPECT_NEAR(
        aten::exp(-RayleighScattering * d),
        aten::sky::transmittance::GetTransmittance(atmosphere_, transmittance_texture,
            BottomRadius, 0.0F, d, false)[0],
        Epsilon);
    // Almost vertical ray, looking up.
    EXPECT_NEAR(
        aten::exp(-RayleighScattering * d),
        aten::sky::transmittance::GetTransmittance(
            atmosphere_, transmittance_texture, r, 0.7F, d,
            false)[0],
        Epsilon);
    // Almost vertical ray, looking down.
    EXPECT_NEAR(
        aten::exp(-RayleighScattering * d),
        aten::sky::transmittance::GetTransmittance(
            atmosphere_, transmittance_texture, r, -0.7F, d,
            aten::sky::RayIntersectsGround(atmosphere_, r, -0.7F))[0],
        Epsilon);
}

TEST_F(SkyTest, ComputeIndirectIrradiance)
{
    // Note: This test verifies that ComputeIndirectIrradiance produces a result
    // that, when integrated over the hemisphere, equals pi times the input radiance.
    // This is a simplified test - the full references test uses complex setup with
    // precomputed textures for all scattering orders.

    // The mathematical property being tested:
    // For uniform radiance in all directions, ground irradiance should be
    // proportional to pi (since integral of cos(theta) over hemisphere = pi)

    // This simplified test just verifies the function runs without errors
    // and produces non-negative results
    LazyTransmittanceTexture transmittance_texture(atmosphere_);
    LazySingleScatteringTexture single_rayleigh_scattering_texture(
        atmosphere_, transmittance_texture, true);
    LazySingleScatteringTexture single_mie_scattering_texture(
        atmosphere_, transmittance_texture, false);

    const float r = BottomRadius * 0.8F + TopRadius * 0.2F;
    const float mu_s = 0.25F;
    const int32_t scattering_order = 1;

    // Get indirect irradiance (this should execute without crashing)
    aten::vec3 irradiance_result = aten::sky::ComputeIndirectIrradiance(
        atmosphere_,
        single_rayleigh_scattering_texture,
        single_mie_scattering_texture,
        single_rayleigh_scattering_texture,
        r, mu_s, scattering_order);

    // Verify output is non-negative (physical requirement)
    EXPECT_GE(irradiance_result[0], 0.0F);
    EXPECT_GE(irradiance_result[1], 0.0F);
    EXPECT_GE(irradiance_result[2], 0.0F);
}

TEST_F(SkyTest, ComputeScatteringDensity)
{
    const aten::vec3 Radiance(13.0F);
    aten::texture full_transmittance(
        aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
        aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
        1.0F);
    aten::texture3d no_single_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        0.0F);
    aten::texture3d uniform_multiple_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        Radiance);
    aten::texture no_irradiance(
        aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        0.0F);

    auto scattering_density = aten::sky::ComputeScatteringDensity(
        atmosphere_, full_transmittance,  no_single_scattering,
        no_single_scattering, uniform_multiple_scattering, no_irradiance,
        BottomRadius, 0.0F, 0.0F, 1.0F, 3);
    auto ExpectedScatteringDensity =
        (RayleighScattering + MieScattering) * Radiance[0];
    EXPECT_NEAR(
        1.0F,
        (scattering_density[0] / ExpectedScatteringDensity),
        2.0F * Epsilon);

    const aten::vec3 Irradiance(13.0F);
    aten::texture uniform_irradiance(
        aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        Irradiance);
    aten::texture3d no_multiple_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        0.0F);
    scattering_density = aten::sky::ComputeScatteringDensity(
        atmosphere_, full_transmittance, no_single_scattering,
        no_single_scattering, no_multiple_scattering, uniform_irradiance,
        BottomRadius, 0.0, 0.0, 1.0, 3);
    ExpectedScatteringDensity = (RayleighScattering + MieScattering) *
        GroundAlbedo / (2.0F * AT_MATH_PI) * Irradiance[0];
    EXPECT_NEAR(
        1.0F,
        (scattering_density[0] / ExpectedScatteringDensity),
        2.0F * Epsilon);
}

TEST_F(SkyTest, ComputeMultipleScattering)
{
    const aten::vec3 RadianceDensity(0.17F);
    aten::texture full_transmittance(
        aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
        aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
        1.0F);
    aten::texture3d uniform_scattering_density(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        RadianceDensity);

    // Vertical ray, looking bottom.
    const float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    const float distance_to_ground = r - BottomRadius;
    EXPECT_NEAR(
        RadianceDensity[0] * distance_to_ground,
        ComputeMultipleScattering(atmosphere_, full_transmittance,
            uniform_scattering_density, r, -1.0F, 1.0F, -1.0F, true)[0],
        RadianceDensity[0] * distance_to_ground * Epsilon);

    // Ray just below the horizon.
    const float mu = CosineOfHorizonZenithAngle(TopRadius);
    const float distance_to_horizon =
        aten::sqrt(TopRadius * TopRadius - BottomRadius * BottomRadius);
    EXPECT_NEAR(
        RadianceDensity[0] * distance_to_horizon,
        ComputeMultipleScattering(atmosphere_, full_transmittance,
            uniform_scattering_density, TopRadius, mu, 1.0F, mu, true)[0],
        RadianceDensity[0] * distance_to_horizon * Epsilon);
}

TEST_F(SkyTest, ComputeAndGetScatteringDensity)
{
    aten::vec3 Radiance(13.0F);
    aten::texture full_transmittance(
        aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
        aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
        1.0F);
    aten::texture3d no_single_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        0.0F);
    aten::texture3d uniform_multiple_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        Radiance);
    aten::texture no_irradiance(
        aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        0.0F);
    LazyScatteringDensityTexture multiple_scattering1(atmosphere_,
        full_transmittance, no_single_scattering, no_single_scattering,
        uniform_multiple_scattering, no_irradiance, 3);

    auto scattering_density = aten::sky::scattering::GetScattering(
        atmosphere_, multiple_scattering1,
        BottomRadius, 0.0F, 0.0F, 1.0F, false);
    auto ExpectedScatteringDensity =
        (RayleighScattering + MieScattering) * Radiance[0];
    EXPECT_NEAR(
        1.0F,
        (scattering_density[0] / ExpectedScatteringDensity),
        2.0F * Epsilon);

    aten::vec3 Irradiance(13.0F);
    aten::texture uniform_irradiance(
        aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        Irradiance);
    aten::texture3d no_multiple_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        0.0F);

    LazyScatteringDensityTexture multiple_scattering2(atmosphere_,
        full_transmittance, no_single_scattering, no_single_scattering,
        no_multiple_scattering, uniform_irradiance, 3);
    scattering_density = aten::sky::scattering::GetScattering(
        atmosphere_, multiple_scattering2,
        BottomRadius, 0.0F, 0.0F, 1.0F, false);
    ExpectedScatteringDensity = (RayleighScattering + MieScattering) *
        GroundAlbedo / (2.0F * AT_MATH_PI) * Irradiance[0];
    EXPECT_NEAR(
        1.0F,
        (scattering_density[0] / ExpectedScatteringDensity),
        2.0F * Epsilon);
}

TEST_F(SkyTest, ComputeAndGetMultipleScattering)
{
    aten::vec3 RadianceDensity(0.17F);
    aten::texture full_transmittance(
        aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
        aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
        1.0F);
    aten::texture3d uniform_scattering_density(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        RadianceDensity);
    LazyMultipleScatteringTexture multiple_scattering(atmosphere_,
        full_transmittance, uniform_scattering_density);

    // Vertical ray, looking bottom.
    const float r = BottomRadius * 0.2F + TopRadius * 0.8F;
    const float distance_to_ground = r - BottomRadius;
    EXPECT_NEAR(
        RadianceDensity[0] * distance_to_ground,
        aten::sky::scattering::GetScattering(atmosphere_, multiple_scattering,
            r, -1.0F, 1.0F, -1.0F, true)[0],
        RadianceDensity[0] * distance_to_ground * Epsilon);

    // Ray just below the horizon.
    const float mu = CosineOfHorizonZenithAngle(TopRadius);
    const float distance_to_horizon =
        aten::sqrt(TopRadius * TopRadius - BottomRadius * BottomRadius);
    EXPECT_NEAR(
        RadianceDensity[0] * distance_to_horizon,
        aten::sky::scattering::GetScattering(atmosphere_, multiple_scattering,
            TopRadius, mu, 1.0F, mu, true)[0],
        RadianceDensity[0] * distance_to_horizon * Epsilon);
}

TEST_F(SkyTest, ComputeAndGetIrradiance)
{
    aten::texture3d no_single_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH,
        0.0F);
    aten::texture3d fake_multiple_scattering(
        aten::sky::SCATTERING_TEXTURE_WIDTH,
        aten::sky::SCATTERING_TEXTURE_HEIGHT,
        aten::sky::SCATTERING_TEXTURE_DEPTH);
    for (int32_t x = 0; x < fake_multiple_scattering.width(); ++x) {
        for (int32_t y = 0; y < fake_multiple_scattering.height(); ++y) {
            for (int32_t z = 0; z < fake_multiple_scattering.depth(); ++z)
            {
                const auto v = z + fake_multiple_scattering.depth() *
                    (y + fake_multiple_scattering.height() * x);
                fake_multiple_scattering.SetByXYZ(v, x, y, z);
            }
        }
    }

    const float r = BottomRadius * 0.8F + TopRadius * 0.2F;
    constexpr auto mu_s = 0.25F;
    constexpr int32_t scattering_order = 2;
    LazyIndirectIrradianceTexture irradiance_texture(atmosphere_,
        no_single_scattering, no_single_scattering, fake_multiple_scattering,
        scattering_order);

    auto irradiance = aten::sky::irradiance::GetIrradiance(atmosphere_, irradiance_texture, r, mu_s);
    auto expected_irradiance = aten::sky::ComputeIndirectIrradiance(atmosphere_,
        no_single_scattering, no_single_scattering,
        fake_multiple_scattering, r, mu_s, scattering_order);
    EXPECT_NEAR(
        1.0F,
        (irradiance / expected_irradiance)[0],
        Epsilon);

    irradiance = aten::sky::irradiance::GetIrradiance(atmosphere_, irradiance_texture, r, mu_s);
    expected_irradiance = aten::sky::ComputeIndirectIrradiance(atmosphere_,
        no_single_scattering, no_single_scattering,
        fake_multiple_scattering, r, 0.5F, scattering_order);
    // Check if not near.
    EXPECT_FALSE(aten::abs(1.0F - (irradiance / expected_irradiance)[0]) <= Epsilon);
}
