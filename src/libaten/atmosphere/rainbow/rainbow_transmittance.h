#pragma once

#include "atmosphere/rainbow/rainbow_constants.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_types.h"
#include "atmosphere/sky/unit_quantity.h"

#include "math/aabb.h"
#include "image/texture_3d.h"

namespace aten::rainbow
{
    inline AT_DEVICE_API float ComputeOpticalDepthBasedOnAabbCoveredSphere(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const float r,
        const float mu)
    {
        const auto profile = atmosphere.mie_density;

        const auto Rg = 0.1F;
        const auto Rt = box.getDiagonalLenght() + Rg;

        // TODO:
        // GetRMuFromTransmittanceTextureUv の内部で、atmosphere.bottom_radius を r に加算しているが
        // float の精度の問題で、r から atmosphere.bottom_radius を減算しても、加算前の値が得られないことがある.
        // そこで、ここでは、r_o が Rt を越えないように clamp する.
        const auto r_o = aten::clamp(r - atmosphere.bottom_radius + Rg, Rg, Rt);

        auto dummy_atmosphere = atmosphere;
        dummy_atmosphere.top_radius = Rt;
        dummy_atmosphere.bottom_radius = Rg;

        const auto optical_length =sky::ComputeOpticalLengthToTopAtmosphereBoundary(
            dummy_atmosphere,
            profile,
            r_o,
            mu);
        return optical_length;
    }

    inline AT_DEVICE_API void GetRMuFromTransmittanceTextureUv(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const aten::vec2& uv,
        float& r,
        float& mu)
    {
        const auto Rg = 0.1F;
        const auto Rt = box.getDiagonalLenght() + Rg;

        auto dummy_atmosphere = atmosphere;
        dummy_atmosphere.top_radius = Rt;
        dummy_atmosphere.bottom_radius = Rg;

        sky::GetRMuFromTransmittanceTextureUv(dummy_atmosphere, uv, r, mu);

        r = r - Rg + atmosphere.bottom_radius;
    }

    inline AT_DEVICE_API vec2 GetTransmittanceTextureUvFromRMu(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const float r,
        const float mu)
    {
        const auto Rg = 0.1F;
        const auto Rt = box.getDiagonalLenght() + Rg;

        auto dummy_atmosphere = atmosphere;
        dummy_atmosphere.top_radius = Rt;
        dummy_atmosphere.bottom_radius = Rg;

        const auto r_tmp = r - atmosphere.bottom_radius + Rg;

        return sky::GetTransmittanceTextureUvFromRMu(dummy_atmosphere, r_tmp, mu);
    }

    inline AT_DEVICE_API aten::tuple<float, float> ComputeRMu(
        const aten::vec3& earth_center,
        const aten::vec3& point,
        const aten::vec3& view_dir)
    {
        auto z = point - earth_center;
        const auto r = length(z);

        z /= r;
        const auto mu = dot(z, view_dir);

        return aten::make_tuple(r, mu);
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceInRainVolume(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const aten::vec3& earth_center,
        const sky::texture2d& transmission_rain_tex,
        const aten::vec3& base_point,
        const aten::vec3& view_dir,
        const float d)
    {
        const auto Rg = 0.1F;
        const auto Rt = box.getDiagonalLenght() + Rg;

        auto dummy_atmosphere = atmosphere;
        dummy_atmosphere.bottom_radius = Rg;
        dummy_atmosphere.top_radius = Rt;

        float r, mu;

        aten::tie(r, mu) = ComputeRMu(earth_center, base_point, view_dir);

        // TODO:
        // GetRMuFromTransmittanceTextureUv の内部で、atmosphere.bottom_radius を r に加算しているが
        // float の精度の問題で、r から atmosphere.bottom_radius を減算しても、加算前の値が得られないことがある.
        // そこで、ここでは、r_o が Rt を越えないように clamp する.
        const auto r_o = aten::clamp(r - atmosphere.bottom_radius + Rg, Rg, Rt);

        const auto transmittance = sky::transmittance::GetTransmittance(
            dummy_atmosphere,
            transmission_rain_tex,
            r_o, mu, d,
            false);

        return aten::vmin(transmittance, 1.0F);
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceInRainVolumeBetweenTwoPoints(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const aten::vec3& earth_center,
        const sky::texture2d& transmission_rain_tex,
        const aten::vec3& point_1,
        const aten::vec3& point_2)
    {
        const auto dir{
            normalize(point_2 - point_1)
        };

        const auto d = length(point_2 - point_1);

        const auto transmittance = GetTransmittanceInRainVolume(
            atmosphere,
            box,
            earth_center,
            transmission_rain_tex,
            point_1, dir, d);

        return aten::vmin(transmittance, 1.0F);
    }

    inline AT_DEVICE_API aten::vec3 GetSkyTransmittance(
        const sky::AtmosphereParameters& atmosphere,
        const aten::vec3& earth_center,
        const sky::texture2d& transmission_tex,
        const aten::vec3& base_point,
        const aten::vec3& view_dir,
        const float d)
    {
        float r, mu;
        aten::tie(r, mu) = ComputeRMu(earth_center, base_point, view_dir);

        const bool is_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

        if (is_intersects_ground) {
            const auto transmittance{
                sky::transmittance::GetTransmittance(
                    atmosphere,
                    transmission_tex,
                    r, mu, d,
                    is_intersects_ground)
            };
            return transmittance;
        }
        else {
            const auto transmittance{
                sky::transmittance::GetTransmittanceToTopAtmosphereBoundary(
                    atmosphere,
                    transmission_tex,
                    r, mu)
            };
            return transmittance;
        }
    }

    // TODO
    inline float ComputeExtinctionInRain(
        const float intensity_rainfall_rate)    // [mm/h]
    {
        // NOTE:
        // Compute as km.

        constexpr auto dD = Length::as(A_STEP, MeterUnit::mm);

        float droplet_diameter_mm = Length::as(A_MIN, MeterUnit::mm);

        float extinction = 0.0F;

        for (int32_t a = 0; a < A_WIDTH; a++) {
            const auto droplet_diameter_m = Length::from(droplet_diameter_mm, MeterUnit::mm, MeterUnit::km);

            // TODO
            // The following code should be standardized.
            constexpr auto N0 = 8000.0F * 10e3F * 10e8F;    // [m^-3mm^-1] -> [m^-4] -> [km^-4]
            const auto lambda = 4.1F * 10e3F * 10e2F * aten::pow(intensity_rainfall_rate, -0.21F);
            const auto droplet_distrib = N0 * aten::exp(-lambda * droplet_diameter_m);
            const auto droplet_radius_m = droplet_diameter_m * 0.5F;
            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_m * droplet_radius_m;

            const auto extinction_i = EXTINCTION_EFFICIENT_IN_RAIN_VOLUME * droplet_cross_sectional_area * droplet_distrib;

            droplet_diameter_mm += dD;

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            float weight_i = a == 0 || a == A_WIDTH - 1 ? 0.5F : 1.0F;
            extinction += extinction_i * weight_i * dD;
        }

        return extinction;
    }

    inline AT_DEVICE_API aten::tuple<float, float> ComputeRMuS(
        const aten::vec3& curr_point,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center)
    {
        auto z = curr_point - earth_center;
        const auto r = length(z);

        z /= r;
        const auto mu_s = dot(z, sun_direction);

        return aten::make_tuple(r, mu_s);
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceToSun(
        const sky::AtmosphereParameters& atmosphere,
        const sky::texture2d& transmittance_texture,
        const aten::vec3& curr_point,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center)
    {
        float r, mu_s;
        aten::tie(r, mu_s) = ComputeRMuS(curr_point, sun_direction, earth_center);

        const auto transmittance = sky::transmittance::GetTransmittanceToSun(
            atmosphere,
            transmittance_texture,
            r, mu_s);

        return transmittance;
    }
}
