#pragma once

#include <gtest/gtest.h>

#include "atmosphere/rainbow/rainbow_transmittance.h"

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


class RainbowTest : public testing::Test {
protected:
    aten::sky::AtmosphereParameters atmosphere_;
    aten::aabb rain_volume_;
    aten::vec3 camera_pos_{
        0.0F,
        aten::Length::as(2.0F, aten::MeterUnit::km),
        0.0F,
    };

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

        {
            constexpr aten::Length RainVolumeWidth = 4.0_km;
            constexpr aten::Length RainVolumeHeight = 4.0_km;
            constexpr aten::Length RainVolumeDepth = 4.0_km;

            aten::vec3 rain_volume_min{
                camera_pos_.x - RainVolumeWidth.as(aten::MeterUnit::km) * 0.5f,
                0.0F,
                camera_pos_.z - 1.0F - RainVolumeDepth.as(aten::MeterUnit::km),
            };
            aten::vec3 rain_volume_max{
                camera_pos_.x + RainVolumeWidth.as(aten::MeterUnit::km) * 0.5f,
                rain_volume_min.y + RainVolumeHeight.as(aten::MeterUnit::km),
                camera_pos_.z - 1.0F,
            };

            rain_volume_.init(
                rain_volume_min,
                rain_volume_max);
        }
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
};
