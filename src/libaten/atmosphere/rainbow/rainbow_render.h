#pragma once

#include "atmosphere/rainbow/rainbow_defs.h"
#include "atmosphere/rainbow/rainbow_compute.h"
#include "atmosphere/rainbow/rainbow_constants.h"
#include "atmosphere/rainbow/rainbow_transmittance.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_render.h"
#include "atmosphere/sky/sky_types.h"
#include "atmosphere/sky/unit_quantity.h"

#include "math/aabb.h"
#include "image/texture_3d.h"

namespace aten::rainbow
{
    inline AT_DEVICE_API float SampleUniformDropletRadius(const float u)
    {
        return DROPLET_SAMPLE_RADIUS_MIN
            + (DROPLET_SAMPLE_RADIUS_MAX - DROPLET_SAMPLE_RADIUS_MIN) * aten::saturate(u);
    }

    inline AT_DEVICE_API float ComputeInverseErfWinitzki(const float z)
    {
        constexpr float a = 0.147F;
        const float x = aten::clamp(z, -0.999999F, 0.999999F);
        if (aten::abs(x) <= AT_MATH_EPSILON) {
            return 0.0F;
        }

        const float l = aten::log(1.0F - x * x);
        const float w = 2.0F / (AT_MATH_PI * a) + l * 0.5F;
        const float inner_sqrt = aten::sqrt(aten::max(0.0F, w * w - l / a));
        return aten::sign(x) * aten::sqrt(aten::max(0.0F, inner_sqrt - w));
    }

    inline AT_DEVICE_API float ComputeInverseNormalDistributionCDF(
        const float u,
        const float mu,
        const float sigma)
    {
        constexpr float SQRT_2 = 1.41421356237309504880F;
        const float clamped_u = aten::clamp(u, 1e-6F, 1.0F - 1e-6F);
        return mu + sigma * SQRT_2 * ComputeInverseErfWinitzki(2.0F * clamped_u - 1.0F);
    }

    inline AT_DEVICE_API float SampleNormalDropletRadius(const float u)
    {
        return ComputeInverseNormalDistributionCDF(
            u,
            DROPLET_SAMPLE_RADIUS_MEAN,
            DROPLET_SAMPLE_RADIUS_SIGMA);
    }

    inline AT_DEVICE_API float SampleMarshallPalmerDropletRadius(
        const float u,
        const float intensity_rainfall_rate)
    {
        const float lambda = ComputeMarshallPalmerDropletSizeDistributionLambda(intensity_rainfall_rate);

        const float d_min_mm = Length::as(DROPLET_SAMPLE_RADIUS_MIN * 2.0F, MeterUnit::mm);
        const float d_max_mm = Length::as(DROPLET_SAMPLE_RADIUS_MAX * 2.0F, MeterUnit::mm);

        const float c0 = aten::exp(-lambda * d_min_mm);
        const float c1 = aten::exp(-lambda * d_max_mm);

        const float d_mm = -aten::log(c0 - aten::saturate(u) * (c0 - c1)) / lambda;

        return Length::from(d_mm * 0.5F, MeterUnit::mm, MeterUnit::m);
    }

    inline AT_DEVICE_API float GetMarshallPalmerDropletDiameterPDFTruncated(
        const float droplet_diameter, // [m]
        const float intensity_rainfall_rate)
    {
        const float lambda = ComputeMarshallPalmerDropletSizeDistributionLambda(intensity_rainfall_rate);

        const float D_mm = Length::as(droplet_diameter, MeterUnit::mm);

        const float D_min_mm = Length::as(DROPLET_SAMPLE_RADIUS_MIN * 2.0F, MeterUnit::mm);
        const float D_max_mm = Length::as(DROPLET_SAMPLE_RADIUS_MAX * 2.0F, MeterUnit::mm);

        if (D_mm < D_min_mm || D_mm > D_max_mm) {
            return 0.0F;
        }

        const float norm = aten::exp(-lambda * D_min_mm) - aten::exp(-lambda * D_max_mm);

        return lambda * aten::exp(-lambda * D_mm) / norm;
    }

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
        const aten::sky::texture3d& airy_func_res_tex)
    {
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

        aten::vec3 uvw{
            aten::saturate(((theta - THETA_MIN) / THETA_STEP + 0.5F) / THETA_WIDTH),
            0.0F,   // compute from wavelength while the integral calculation.
            0.0F,   // compute from droplet radius while the integral calculation.
        };

        const auto solar_radiance{ sky::GetSolarRadiance(atmosphere) };

        // [nm] -> [m].
        constexpr std::array visible_wavelength = {
            sky::LambdaR * 1e-9F,
            sky::LambdaG * 1e-9F,
            sky::LambdaB * 1e-9F,
        };

        const auto p_max = 1.0F - ComputeMarshallPalmerDropletSizeDistributionFactor(A_MAX * 2.0F, intensity_rainfall_rate);
        const auto p_min = 1.0F - ComputeMarshallPalmerDropletSizeDistributionFactor(A_MIN * 2.0F, intensity_rainfall_rate);

        aten::vec3 rainbow_radiance{ 0.0F };
        float optical_length_in_rain_volume = 0.0F;

        aten::vec3 rainbow_radiance_tmp{ 0.0F };

        const aten::vec3& move_dir = view_dir;

        for (size_t i = 0; i <= SAMPLE_COUNT; i++) {
            AT_ASSERT(rain_volume.isIn(curr_point));

            const auto d_i = i * dt;

#ifdef ENABLE_PRECOMPUTE_DROPLET_RADIUS
            // Sample from pre computed textures.
            float droplet_radius, rain_weight;
            aten::tie(droplet_radius, rain_weight) = GetDropletRadiusAndRainDensityWeightFromPreComputeTexture(droplet_radius_tex, curr_point, rain_volume);
#else
            float droplet_size_pdf = 1.0F;

            auto u = sampler.nextSample();

#if 0
            const auto droplet_radius = SampleUniformDropletRadius(u);
#elif 0
            const auto droplet_radius = SampleNormalDropletRadius(u);
#else
            const auto droplet_radius = SampleMarshallPalmerDropletRadius(u, intensity_rainfall_rate);
            droplet_size_pdf = GetMarshallPalmerDropletDiameterPDFTruncated(
                droplet_radius * 2.0F,
                intensity_rainfall_rate);
#endif

            const auto rain_density = ComputeMarshallPalmerDropletSizeDistribution(
                2.0F * droplet_radius,  // diameter
                intensity_rainfall_rate);
            const auto rain_weight = droplet_size_pdf > 0.0F
                ? rain_density / droplet_size_pdf
                : 0.0F;
#endif

            uvw.z = aten::saturate(((droplet_radius - A_MIN) / A_STEP + 0.5F) / A_WIDTH);

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

            aten::vec3 rainbow_intensity;

#ifdef ENABLE_FULL_SPECTRAL_RAINBOW
            uvw.y = 0.5F / WAVELENGTH_WIDTH;
            rainbow_intensity = sky::SampleTexture3D(airy_func_res_tex, uvw);
#else
            for (size_t n = 0; n < visible_wavelength.size(); n++) {
                const auto wavelength = visible_wavelength[n];
                uvw.y = aten::saturate(((wavelength - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F) / WAVELENGTH_WIDTH);
                rainbow_intensity[n] =  GetAiryFunctionValue(airy_func_res_tex, uvw);;
            }
#endif

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
                * solar_radiance
                * rainbow_intensity
                * rain_weight
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
