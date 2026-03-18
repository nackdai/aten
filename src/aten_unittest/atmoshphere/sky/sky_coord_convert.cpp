#include "sky_test_fixture.h"

TEST_F(SkyTest, GetTextureCoordFromUnitRange)
{
    EXPECT_NEAR(0.5F / 10.0F, aten::sky::GetTextureCoordFromUnitRange(0.0F, 10), Epsilon);
    EXPECT_NEAR(9.5F / 10.0F, aten::sky::GetTextureCoordFromUnitRange(1.0F, 10), Epsilon);
}

TEST_F(SkyTest, GetUnitRangeFromTextureCoord)
{
    EXPECT_NEAR(0.0F, aten::sky::GetUnitRangeFromTextureCoord(0.5F / 10.0F, 10), Epsilon);
    EXPECT_NEAR(1.0F, aten::sky::GetUnitRangeFromTextureCoord(9.5F / 10.0F, 10), Epsilon);

    EXPECT_NEAR(1.0F / 3.0F,
        aten::sky::GetUnitRangeFromTextureCoord(
            aten::sky::GetTextureCoordFromUnitRange(1.0F / 3.0F, 10), 10),
        Epsilon);
}

TEST_F(SkyTest, GetTransmittanceTextureUvFromRMu)
{
    aten::vec2 uv = aten::sky::GetTransmittanceTextureUvFromRMu(
        atmosphere_, TopRadius, 1.0F);
    EXPECT_NEAR(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH, uv.x, Epsilon);
    EXPECT_NEAR(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT, uv.y, Epsilon);

    uv = aten::sky::GetTransmittanceTextureUvFromRMu(
        atmosphere_, TopRadius,
        -aten::sqrt(1.0F - (BottomRadius / TopRadius) * (BottomRadius / TopRadius)));
    EXPECT_NEAR(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH, uv.x, Epsilon);
    EXPECT_NEAR(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT, uv.y, Epsilon);

    uv = aten::sky::GetTransmittanceTextureUvFromRMu(
        atmosphere_, BottomRadius, 1.0F);
    EXPECT_NEAR(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH, uv.x, Epsilon);
    EXPECT_NEAR(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT, uv.y, Epsilon);

    uv = aten::sky::GetTransmittanceTextureUvFromRMu(
        atmosphere_, BottomRadius, 0.0F);
    EXPECT_NEAR(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH, uv.x, Epsilon);
    EXPECT_NEAR(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT, uv.y, Epsilon);
}

TEST_F(SkyTest, GetRMuFromTransmittanceTextureUv)
{
    float r;
    float mu;
    aten::sky::GetRMuFromTransmittanceTextureUv(
        atmosphere_,
        aten::vec2(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                   1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        r, mu);
    EXPECT_NEAR(TopRadius, r, 1.0F);
    EXPECT_NEAR(1.0F, mu, Epsilon);

    aten::sky::GetRMuFromTransmittanceTextureUv(
        atmosphere_,
        aten::vec2(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                   1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        r, mu);
    EXPECT_NEAR(TopRadius, r, 1.0F);
    EXPECT_NEAR(
        -aten::sqrt(1.0F - (BottomRadius / TopRadius) * (BottomRadius / TopRadius)),
        mu,
        Epsilon);

    aten::sky::GetRMuFromTransmittanceTextureUv(
        atmosphere_,
        aten::vec2(0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                   0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        r, mu);
    EXPECT_NEAR(BottomRadius, r, 1.0F);
    EXPECT_NEAR(1.0F, mu, Epsilon);

    aten::sky::GetRMuFromTransmittanceTextureUv(
        atmosphere_,
        aten::vec2(1.0F - 0.5F / aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                   0.5F / aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        r, mu);
    EXPECT_NEAR(BottomRadius, r, 1.0F);
    EXPECT_NEAR(0.0F, mu, Epsilon);

    aten::sky::GetRMuFromTransmittanceTextureUv(
        atmosphere_,
        aten::sky::GetTransmittanceTextureUvFromRMu(atmosphere_,
            BottomRadius * 0.2F + TopRadius * 0.8F, 0.25F),
        r, mu);
    EXPECT_NEAR(BottomRadius * 0.2F + TopRadius * 0.8F, r, 1.0F);
    EXPECT_NEAR(0.25F, mu, Epsilon);
}

TEST_F(SkyTest, GetIrradianceTextureUvFromRMuS)
{
    EXPECT_NEAR(
        0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        aten::sky::GetIrradianceTextureUvFromRMuS(
            atmosphere_, BottomRadius, 0.0F).y,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        aten::sky::GetIrradianceTextureUvFromRMuS(
            atmosphere_, TopRadius, 0.0F).y,
        Epsilon);

    EXPECT_NEAR(
        0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::GetIrradianceTextureUvFromRMuS(
            atmosphere_, BottomRadius, -1.0F).x,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
        aten::sky::GetIrradianceTextureUvFromRMuS(
            atmosphere_, BottomRadius, 1.0F).x,
        Epsilon);
}

TEST_F(SkyTest, GetRMuSFromIrradianceTextureUv)
{
    float r;
    float mu_s;
    aten::sky::GetRMuSFromIrradianceTextureUv(
        atmosphere_,
        aten::vec2(0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                   0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT),
        r, mu_s);
    EXPECT_NEAR(BottomRadius, r, 1.0F);

    aten::sky::GetRMuSFromIrradianceTextureUv(
        atmosphere_,
        aten::vec2(0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                   1.0F - 0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT),
        r, mu_s);
    EXPECT_NEAR(TopRadius, r, 1.0F);

    aten::sky::GetRMuSFromIrradianceTextureUv(
        atmosphere_,
        aten::vec2(0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                   0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT),
        r, mu_s);
    EXPECT_NEAR(-1.0F, mu_s, Epsilon);

    aten::sky::GetRMuSFromIrradianceTextureUv(
        atmosphere_,
        aten::vec2(1.0F - 0.5F / aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                   0.5F / aten::sky::IRRADIANCE_TEXTURE_HEIGHT),
        r, mu_s);
    EXPECT_NEAR(1.0F, mu_s, Epsilon);
}

TEST_F(SkyTest, GetRMuMuSNuFromScatteringTextureUvwz)
{
    float r;
    float mu;
    float mu_s;
    float nu;
    bool ray_r_mu_intersects_ground;

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(BottomRadius, r, 1.0F);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(TopRadius, r, 1.0F);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE + Epsilon,
                   0.5F),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    const float mu_horizon = CosineOfHorizonZenithAngle(r);
    EXPECT_NEAR(mu_horizon, mu, Epsilon);
    EXPECT_LE(mu, mu_horizon);
    EXPECT_TRUE(ray_r_mu_intersects_ground);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE - Epsilon,
                   0.5F),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(mu_horizon, mu, 5.0F * Epsilon);
    EXPECT_GE(mu, mu_horizon);
    EXPECT_FALSE(ray_r_mu_intersects_ground);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(-1.0F, mu_s, Epsilon);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(1.0F, mu_s, Epsilon);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(0.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(-1.0F, nu, Epsilon);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::vec4(1.0F,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
                   0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(1.0F, nu, Epsilon);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(atmosphere_,
            BottomRadius, -1.0F, 1.0F, -1.0F, true),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(BottomRadius, r, 1.0F);
    EXPECT_NEAR(-1.0F, mu, Epsilon);
    EXPECT_NEAR(1.0F, mu_s, Epsilon);
    EXPECT_NEAR(-1.0F, nu, Epsilon);
    EXPECT_TRUE(ray_r_mu_intersects_ground);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(atmosphere_,
            TopRadius, -1.0F, 1.0F, -1.0F, true),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR(TopRadius, r, 1.0F);
    EXPECT_NEAR(-1.0F, mu, Epsilon);
    EXPECT_NEAR(1.0F, mu_s, Epsilon);
    EXPECT_NEAR(-1.0F, nu, Epsilon);
    EXPECT_TRUE(ray_r_mu_intersects_ground);

    aten::sky::GetRMuMuSNuFromScatteringTextureUvwz(atmosphere_,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(atmosphere_,
            (BottomRadius + TopRadius) / 2.0F, 0.2F, 0.3F, 0.4F, false),
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    EXPECT_NEAR((BottomRadius + TopRadius) / 2.0F, r, 1.0F);
    EXPECT_NEAR(0.2F, mu, Epsilon);
    EXPECT_NEAR(0.3F, mu_s, Epsilon);
    EXPECT_NEAR(0.4F, nu, Epsilon);
    EXPECT_FALSE(ray_r_mu_intersects_ground);
}

TEST_F(SkyTest, GetScatteringTextureUvwzFromRMuMuSNu)
{
    EXPECT_NEAR(
        0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, BottomRadius, 0.0F, 0.0F, 0.0F, false).w,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_R_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, TopRadius, 0.0F, 0.0F, 0.0F, false).w,
        Epsilon);

    const float r = (TopRadius + BottomRadius) / 2.0F;
    const float mu = CosineOfHorizonZenithAngle(r);
    EXPECT_NEAR(
        0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, r, mu, 0.0F, 0.0F, true).z,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_MU_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, r, mu, 0.0F, 0.0F, false).z,
        Epsilon);
    EXPECT_TRUE(aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere_, r, -1.0F, 0.0F, 0.0F, true).z < 0.5F);
    EXPECT_TRUE(aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere_, r, 1.0F, 0.0F, 0.0F, false).z > 0.5F);

    EXPECT_NEAR(
        0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, BottomRadius, 0.0F, -1.0F, 0.0F, false).y,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, BottomRadius, 0.0F, 1.0F, 0.0F, false).y,
        Epsilon);

    EXPECT_NEAR(
        0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, TopRadius, 0.0F, -1.0F, 0.0F, false).y,
        Epsilon);
    EXPECT_NEAR(
        1.0F - 0.5F / aten::sky::SCATTERING_TEXTURE_MU_S_SIZE,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, TopRadius, 0.0F, 1.0F, 0.0F, false).y,
        Epsilon);

    EXPECT_NEAR(
        0.0F,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, BottomRadius, 0.0F, 0.0F, -1.0F, false).x,
        Epsilon);
    EXPECT_NEAR(
        1.0F,
        aten::sky::GetScatteringTextureUvwzFromRMuMuSNu(
            atmosphere_, BottomRadius, 0.0F, 0.0F, 1.0F, false).x,
        Epsilon);
}
