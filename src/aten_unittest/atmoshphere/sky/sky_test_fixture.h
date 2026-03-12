#pragma once

#include <gtest/gtest.h>

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_coord_convert.h"
#include "atmosphere/sky/sky_params.h"

constexpr float km = 1000.0F;
constexpr float watt_per_square_meter_per_nm = 1.0F;

constexpr float Epsilon = 1e-3F;

constexpr float SolarIrradiance = 123.0F * watt_per_square_meter_per_nm;
constexpr float BottomRadius = 1000.0F * km;
constexpr float TopRadius = 1500.0F * km;
constexpr float ScaleHeight = 60.0F * km;
constexpr float RayleighScaleHeight = 60.0F * km;
constexpr float MieScaleHeight = 30.0F * km;
constexpr float RayleighScattering = 0.001F / km;
constexpr float MieScattering = 0.0015F / km;
constexpr float MieExtinction = 0.002F / km;
constexpr float GroundAlbedo = 0.1F;

class SkyTest : public testing::Test {
protected:
    aten::sky::AtmosphereParameters atmosphere_;

    void SetUp() override
    {
        memset(&atmosphere_, 0, sizeof(atmosphere_));
        atmosphere_.solar_irradiance[0] = SolarIrradiance;
        atmosphere_.bottom_radius = BottomRadius;
        atmosphere_.top_radius = TopRadius;
        atmosphere_.rayleigh_density.layers[1] = aten::sky::DensityProfileLayer{
            0.0F, 1.0F, -1.0F / RayleighScaleHeight, 0.0F, 0.0F };
        atmosphere_.rayleigh_scattering[0] = RayleighScattering;
        atmosphere_.mie_density.layers[1] = aten::sky::DensityProfileLayer{
            0.0F, 1.0F, -1.0F / MieScaleHeight, 0.0F, 0.0F };
        atmosphere_.mie_scattering[0] = MieScattering;
        atmosphere_.mie_extinction[0] = MieExtinction;
        atmosphere_.ground_albedo[0] = GroundAlbedo;
        atmosphere_.mu_s_min = -1.0;
    }

    void SetUniformAtmosphere()
    {
        atmosphere_.rayleigh_density.layers[0] = aten::sky::DensityProfileLayer{};
        atmosphere_.rayleigh_density.layers[1] = aten::sky::DensityProfileLayer{
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F };
        atmosphere_.mie_density.layers[0] = aten::sky::DensityProfileLayer{};
        atmosphere_.mie_density.layers[1] = aten::sky::DensityProfileLayer{
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F };
        //atmosphere_.absorption_density.layers[0] = aten::sky::DensityProfileLayer{};
        //atmosphere_.absorption_density.layers[1] = aten::sky::DensityProfileLayer{};
    }

    void RemoveAerosols()
    {
        atmosphere_.mie_scattering[0] = 0.0F;
        atmosphere_.mie_extinction[0] = 0.0F;
    }

    /**
     * Computes the cosine of the zenith angle to the horizon at radius r.
     * * Geometrically, this represents the boundary between rays that hit the
     * ground and rays that escape into the atmosphere.
     *
     * @param r The distance from the planet center to the current point.
     * @return The cosine of the horizon zenith angle (negative value).
     *
     * * Derivation:
     * Let theta be the angle such that cos(theta) = kBottomRadius / r.
     * The horizon zenith angle is 90 degrees + theta.
     * Using cos(90 + theta) = -sin(theta) and sin = sqrt(1 - cos^2), we get:
     * Result = -sqrt(1.0 - (kBottomRadius / r)^2)
     */
    float CosineOfHorizonZenithAngle(const float r)
    {
        AT_ASSERT(r >= BottomRadius);
        return -aten::sqrt(1.0F - (BottomRadius / r) * (BottomRadius / r));
    }
};
