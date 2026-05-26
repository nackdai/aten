#pragma once

#include "atmosphere/rainbow/rainbow_compute.h"

namespace aten::rainbow
{
    inline AT_DEVICE_API aten::vec3 AdvanceRainVolumeIntegral(
        aten::sampler& sampler,
        const sky::AtmosphereParameters& atmosphere,
        const aten::sky::texture2d& transmittance_texture,
        const aten::sky::texture2d& transmittance_in_rain_volume_texture,
        const aten::sky::texture3d& droplet_radius_tex,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center, // [km]
        const aten::vec3& camera_pos,   // [km]
        const aten::vec3& view_dir,
        const aten::aabb& rain_volume,  // [km x km x km]
        const float intensity_rainfall_rate,    // [mm/h]
        const aten::sky::texture2d& spectrum_srgb_tex)
    {
        (void)droplet_radius_tex;
        (void)intensity_rainfall_rate;

        // If the view direction is the same as sun direction, the rainbow doesn't appear.
        const bool is_same_direction = dot(sun_direction, view_dir) > 0.0F;
        if (is_same_direction) {
            return aten::vec3(0.0F);
        }

        // TODO
        // そもそも、太陽が地球の下に隠れて見えないなどについては、
        // GetTransmittanceToSun など sky 側で対応済みなので、それを利用する.
        // ただ、その場合に太陽を点ではなく円盤としているので、太陽の扱いは円盤にすること.

        const float theta = aten::acos(dot(sun_direction, -view_dir));
        if (theta < THETA_MIN || theta >= THETA_MAX) {
            // 主虹、副虹の範囲内に収まらないので、虹が見えない.
            return aten::vec3(0.0F);
        }

        aten::vec3 curr_point;

        float t0, t1;
        aten::tie(t0, t1) = rain_volume.GetHitT(aten::ray(camera_pos, view_dir), AT_MATH_EPSILON, AT_MATH_INF);

        const auto is_hit = t0 <= t1;

        if (rain_volume.isIn(camera_pos)) {
            curr_point = camera_pos;
        }
        else if (is_hit) {
            curr_point = camera_pos + t0 * view_dir;
        }
        else {
            return aten::vec3(0.0F);
        }

        const auto start_pos_in_rain_volume { curr_point };

        constexpr auto SAMPLE_COUNT = 100;

        const auto box_distance_along_with_view_dir = t1 - t0;

        // Not to out of the rain volume at the last step, multiply by 0.99.
        const auto dt = box_distance_along_with_view_dir / SAMPLE_COUNT * 0.99F;

        aten::vec2 uv{
            aten::saturate(((theta - THETA_MIN) / THETA_STEP + 0.5F) / THETA_WIDTH),
            0.0F,   // compute from droplet radius while the integral calculation.
        };

        const auto solar_radiance{ sky::GetSolarRadiance(atmosphere) };

        aten::vec3 rainbow_radiance{ 0.0F };
        float optical_length_in_rain_volume = 0.0F;

        aten::vec3 rainbow_radiance_tmp{ 0.0F };

        const aten::vec3& move_dir = view_dir;

        for (size_t i = 0; i <= SAMPLE_COUNT; i++) {
            AT_ASSERT(rain_volume.isIn(curr_point));

            const auto d_i = i * dt;

            // Current point is in rain volume box. In this case, t0 is always zero. So, we adopt t1.
            aten::tie(t0, t1) = rain_volume.GetHitT(aten::ray(curr_point, sun_direction), AT_MATH_EPSILON, AT_MATH_INF);
            const aten::vec3 boundary_point_to_sun_in_rain_volume{ curr_point + t1 * sun_direction };

            // Transmittance from the current point to sun only within the rain volume.
            const auto transmittance_to_sun_in_rain_volume = GetTransmittanceInRainVolumeBetweenTwoPoints(
                atmosphere, rain_volume,
                earth_center,
                transmittance_in_rain_volume_texture,
                curr_point, boundary_point_to_sun_in_rain_volume);

            float r, mu_s;
            aten::tie(r, mu_s) = ComputeRMuS(curr_point, sun_direction, earth_center);

            // Get transmittance to sun in the atmosphere boundary.
            // Multiply the transmittance in the rain volume.
            const auto transmittance_to_sun = sky::transmittance::GetTransmittanceToSun(
                atmosphere,
                transmittance_texture,
                r, mu_s) * transmittance_to_sun_in_rain_volume;

            aten::vec3 rainbow_intensity{ 0.0F };

            const float u_droplet_radius = sampler.nextSample();
#if defined(AT_RAINBOW_USE_UNIFORM_DROPLET_RADIUS)
            const float droplet_radius = SampleUniformDropletRadius(u_droplet_radius);
#else
            const float droplet_radius = SampleNormalDropletRadius(u_droplet_radius);
#endif
            uv.y = aten::saturate(((droplet_radius - A_MIN) / A_STEP + 0.5F) / A_WIDTH);

            rainbow_intensity += sky::SampleTexture2D(spectrum_srgb_tex, uv);


            const float droplet_diameter = droplet_radius * 2.0F;
            const float rain_density = ComputeMarshallPalmerDropletSizeDistribution(
                droplet_diameter,
                intensity_rainfall_rate);

            const auto transmittance = GetSkyTransmittance(
                atmosphere, earth_center,
                transmittance_texture,
                curr_point, move_dir, d_i);

            const auto transmittance_in_rain_volume = GetTransmittanceInRainVolume(
                    atmosphere, rain_volume, earth_center,
                    transmittance_in_rain_volume_texture,
                    start_pos_in_rain_volume, move_dir, d_i);

            const auto rainbow_radiance_i{
                transmittance
                * transmittance_in_rain_volume
                * transmittance_to_sun
                * rainbow_intensity
                * rain_density
                * solar_radiance
            };

            curr_point += move_dir * dt;

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            float weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5F : 1.0F;

            rainbow_radiance += rainbow_radiance_i * weight_i * dt;
        }

        return rainbow_radiance;
    }
}
