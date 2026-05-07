#pragma once

#include "atmosphere/rainbow/rainbow_constants.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_render.h"
#include "atmosphere/sky/sky_types.h"
#include "atmosphere/sky/unit_quantity.h"

#include "math/aabb.h"
#include "image/texture_3d.h"

namespace aten::rainbow {
    inline AT_DEVICE_API float ComputeInverseNormalDistributionCDF(
        const float u,
        const float mu,
        const float sigma)
    {
        constexpr auto WinitzkiApproxFactor = 0.147F;

        // TODO
        const auto _u = aten::clamp(u, 0.00001F, 0.99999F);

        // Compute inverse erf from Winitzki approximation.
        const auto z = 2.0F * _u - 1.0F;

        float inv_erf = 0.0F;

        if (z != 0.0F) {
            const auto a = WinitzkiApproxFactor;
            const auto l = aten::log(1 - z * z);
            const auto w = 2.0F / (AT_MATH_PI * a) + l / 2.0F;

            const auto inner_sqrt = aten::sqrt(w * w - l / a);
            const auto x = aten::sqrt(inner_sqrt - w);

            inv_erf = z < 0.0F ? -x : x;
        }

        const auto result = mu + sigma * aten::sqrt(2) * inv_erf;

        return result;
    }

    inline AT_DEVICE_API float ComputeMarshallPalmerDropletSizeDistributionLambda(const float intensity_rainfall_rate)
    {
        const auto lambda = 4.1F * aten::pow(intensity_rainfall_rate, -0.21F);
        return lambda;
    }

    inline AT_DEVICE_API float ComputeMarshallPalmerDropletSizeDistributionFactor(
        const float droplet_diameter,   // [m]
        const float intensity_rainfall_rate)
    {
        // m -> mm.
        const auto D = Length::as(droplet_diameter, MeterUnit::mm);
        const auto lambda = ComputeMarshallPalmerDropletSizeDistributionLambda(intensity_rainfall_rate);
        const auto e = aten::exp(-lambda * D);
        return e;
    }

    // マーシャル・パルマー粒径分布を計算.
    // intensity_rainfall_rate[mm/h]
    // droplet_diameter[m]
    inline AT_DEVICE_API float ComputeMarshallPalmerDropletSizeDistribution(
        const float droplet_diameter,
        const float intensity_rainfall_rate)
    {
        // https://en.wikipedia.org/wiki/Raindrop_size_distribution
        // https://www.atmos.albany.edu/facstaff/rfovell/ATM562/marshall-palmer-1948.pdf

        // 論文だと、N0 = 0.008[cm^-4] で計算している.
        // その場合は、D [cm] で計算する必要がある.

        // [m^-3mm^-1]
        constexpr float N0 = 8000.0F;

        const auto e = ComputeMarshallPalmerDropletSizeDistributionFactor(
            droplet_diameter,
            intensity_rainfall_rate);
        const auto ND = N0 * e;

        return ND;
    }

    inline AT_DEVICE_API float GetDropletDiamterFromMarshallPalmerDropletSizeDistribution(
        const float u,
        const float intensity_rainfall_rate)
    {
        const auto lambda = ComputeMarshallPalmerDropletSizeDistributionLambda(intensity_rainfall_rate);
        const auto D = -1.0F / lambda * aten::log(1.0F - u);
        return D;
    }

    inline AT_DEVICE_API float GetMarshallPalmerDropletSizeDistributionPDF(
        const float droplet_diameter,
        const float intensity_rainfall_rate)
    {
        const auto lambda = 4.1F * aten::pow(intensity_rainfall_rate, -0.21F);

        const auto D = Length::as(droplet_diameter, MeterUnit::mm);
        const auto e = aten::exp(-lambda * D);
        return lambda * e;
    }

    // 水の屈折率(20°C)を計算.
    // wavelength[m] e.g. 660e-9[m]
    inline AT_DEVICE_API float ComputeWaterRefractiveIndex(const float wavelength)
    {
        // https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon-20.0C

        // m -> micro meter.
        const auto x = wavelength * 1e6F;

        const auto n = aten::sqrt(
            1 + 5.684027565e-1F / (1 - 5.101829712e-3F / aten::pow(x, 2))
            + 1.726177391e-1F / (1 - 1.821153936e-2F / aten::pow(x, 2))
            + 2.086189578e-2F / (1 - 2.620722293e-2F / aten::pow(x, 2))
            + 1.130748688e-1F / (1 - 1.069792721e1F / aten::pow(x, 2)));

        return n;
    }

    inline AT_DEVICE_API float ComputeH(const float n)
    {
        // n : 屈折率
        const auto n2 = aten::pow(n, 2);
        const auto h = 9.0F / (4.0F * (n2 - 1.0F)) * aten::sqrt((4.0F - n2) / (n2 - 1.0F));
        return h;
    }

    inline AT_DEVICE_API float ComputeThetaMax(const float n)
    {
        // n : 屈折率
        const auto n2 = aten::pow(n, 2);
        const auto theta_max = 4.0F * aten::asin(aten::sqrt((4.0F - n2) / (3.0F * n2))) - 2.0F * aten::asin(aten::sqrt((4.0F - n2) / 3.0F));
        return theta_max;
    }

    inline AT_DEVICE_API float ComputeZ(
        const float wavelength, // [m]
        const float a,  // [m]
        const float theta,
        const float h,
        const float theta_max)
    {
        // wavelength : 波長
        // a : 雨滴半径
        // theta : 角度(ラジアン)
        // h : 計算されたhの値
        // theta_max : 計算されたtheta_maxの値
        const auto z = aten::pow(48.0F / h, 1.0F / 3.0F)
            * aten::pow(a / wavelength, 2.0F / 3.0F)
            * (theta_max - theta);
        return z;
    }

    inline AT_DEVICE_API float ComputeM(
        const float k,
        const float wavelength, // [m]
        const float a,  // droplet radius. [m]
        const float theta,
        const float h,
        const float theta_max)
    {
        // k : 係数
        // wavelength : 波長
        // a : 半径
        // theta : 角度(ラジアン)
        // h : 計算されたhの値
        // theta_max : 計算されたtheta_maxの値
        const auto epsilon = theta_max - theta;
        const auto cos_eps = aten::cos(epsilon);
        const auto M = 2 * k * aten::pow((3 * a * a * wavelength) / (4 * h * cos_eps), 1.0F / 3.0F);
        return M;
    }

    inline AT_DEVICE_API float ComputeAiryFunctionByIntegral(const float z)
    {
        constexpr auto u_max = 10.0F;
        constexpr auto du = 0.005F;

        float f_z = 0.0F;
        const auto n_steps = static_cast<int32_t>(u_max / du);

        for (int32_t i = 0; i < n_steps; i++) {
            auto u = i * du;

            // 式(2) : cos(pi / 2 * (u ^ 3 - z * u))
            auto phase = (AT_MATH_PI / 2.0F) * (aten::pow(u, 3) - z * u);
            auto val = aten::cos(phase);

            // 台形則で数値積分.
            if (i == 0 || i == n_steps) {
                f_z += 0.5F * val;
            }
            else {
                f_z += val;
            }
        }

        return f_z * du;
    }

    inline AT_DEVICE_API void PreComputeAiryFunction(aten::sky::texture3d& airy_func_tex)
    {
        for (int32_t z = 0; z < A_WIDTH; z++) {
            // Droplet radius. Unit is [m].
            const auto a = A_MIN + z * A_STEP;

            for (int32_t y = 0; y < WAVELENGTH_WIDTH; y++) {
                // Unit is [m].
                const auto wavelength = WAVELENGTH_MIN + y * WAVELENGTH_STEP;

                const auto n = ComputeWaterRefractiveIndex(wavelength);
                const auto h = ComputeH(n);
                const auto theta_max = ComputeThetaMax(n);

                for (int32_t x = 0; x < THETA_WIDTH; x++) {
                    const auto theta = THETA_MIN + x * THETA_STEP;

                    const auto M = ComputeM(1.0F, wavelength, a, theta, h, theta_max);

                    const auto z = ComputeZ(wavelength, a, theta, h, theta_max);
                    const auto f_z = ComputeAiryFunctionByIntegral(z);

                    sky::WriteTexture3D(
                        airy_func_tex,
                        aten::vec3(M * M * f_z * f_z),
                        x, y, z);
                }
            }
        }
    }

    inline AT_DEVICE_API float ComputeAiryFunction(
        int32_t x, int32_t y, int32_t z)
    {
        // Droplet radius. Unit is [m].
        const auto a = A_MIN + z * A_STEP;

        // Unit is [m].
        const auto wavelength = WAVELENGTH_MIN + y * WAVELENGTH_STEP;

        const auto n = ComputeWaterRefractiveIndex(wavelength);
        const auto h = ComputeH(n);
        const auto theta_max = ComputeThetaMax(n);

        const auto theta = THETA_MIN + x * THETA_STEP;

        const auto M = ComputeM(1.0F, wavelength, a, theta, h, theta_max);

        const auto airt_func_z = ComputeZ(wavelength, a, theta, h, theta_max);
        const auto f_z = ComputeAiryFunctionByIntegral(airt_func_z);

        return M * M * f_z * f_z;
    }

    inline AT_DEVICE_API float GetAiryFunctionValue(
        const aten::sky::texture3d& airy_func_tex,
        const aten::vec3& uvw)
    {
        AT_ASSERT(uvw.x >= 0.0F && uvw.x <= 1.0F);
        AT_ASSERT(uvw.y >= 0.0F && uvw.y <= 1.0F);
        AT_ASSERT(uvw.z >= 0.0F && uvw.z <= 1.0F);

        // NOTE:
        // the content value is the same in vec3.
        const auto value = sky::SampleTexture3D(airy_func_tex, uvw);
        return value.x;
    }

    inline AT_DEVICE_API float GetAiryFunctionValue(
        const aten::sky::texture3d& airy_func_tex,
        const float wavelength, // [m]
        const float a,  // droplet radius [m]
        const float theta)
    {
        aten::vec3 uvw{
            ((theta - THETA_MIN) / THETA_STEP + 0.5F) / THETA_WIDTH,
            ((wavelength - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F) / WAVELENGTH_WIDTH,
            ((a - A_MIN) / A_STEP + 0.5F) / A_WIDTH
        };

        return GetAiryFunctionValue(airy_func_tex, uvw);
    }

    inline AT_DEVICE_API float GetDropletRadiusFromPreComputeTexture(
        const aten::sky::texture3d& droplet_radius_tex,
        const aten::vec3& point,
        const aten::aabb& volume)
    {
        if (volume.isEmpty() || !volume.isIn(point)) {
            AT_ASSERT(false);
            return 0.0F;
        }

        // Normalize.
        const auto& min_pos = volume.minPos();
        const auto& max_pos = volume.maxPos();
        const auto range{max_pos - min_pos};

        auto uvw{
            (point - min_pos) / range
        };

        AT_ASSERT(0.0F <= uvw.x && uvw.x <= 1.0F);
        AT_ASSERT(0.0F <= uvw.y && uvw.y <= 1.0F);
        AT_ASSERT(0.0F <= uvw.z && uvw.z <= 1.0F);

        // TODO
        uvw.x = aten::saturate(uvw.x);
        uvw.y = aten::saturate(uvw.y);
        uvw.z = aten::saturate(uvw.z);

        const auto droplet_radius = sky::SampleTexture3D(droplet_radius_tex, uvw);
        return droplet_radius.x;
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceToSun(
        const sky::AtmosphereParameters& atmosphere,
        const sky::texture2d& transmittance_texture,
        const aten::vec3& curr_point,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center)
    {
        auto z = curr_point - earth_center;
        const auto r = length(z);

        z /= r;
        const auto mu_s = dot(z, sun_direction);

        const auto transmittance = sky::transmittance::GetTransmittanceToSun(
            atmosphere,
            transmittance_texture,
            r, mu_s);

        return transmittance;
    }

    inline AT_DEVICE_API aten::vec3 ComputeTransmittanceBetweenTwoPoints(
        const sky::AtmosphereParameters& atmosphere,
        const sky::texture2d& transmittance_texture,
        const aten::vec3& point1, const aten::vec3& point2,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center)
    {
        // transmittance は視点をx、大気上端の交点を i とすると、 exp(-∫_x^i).
        // また、途中の点を y とすると、 exp(-∫_x^i) = exp{-(∫_x^y + ∫_y^i}) = exp(-∫_x^y) * exp(-∫_y^i) となる.
        // つまり、exp(-∫_x^y) = exp(-∫_x^i) / exp(-∫_y^i) となるので、
        // transmittance(x,y) = transmittance(x,i) / transmittance(y,i) となる.
        // x ----- y ----- i

        const auto transmittance_from_point1 = GetTransmittanceToSun(atmosphere, transmittance_texture, point1, sun_direction, earth_center);
        const auto transmittance_from_point2 = GetTransmittanceToSun(atmosphere, transmittance_texture, point2, sun_direction, earth_center);

        const auto transmittance_between_points = transmittance_from_point1 / transmittance_from_point2;

        return transmittance_between_points;
    }

    // TODO
    inline AT_DEVICE_API float ComputeExtinctionInRain(
        const float intensity_rainfall_rate)    // [mm/h]
    {
        // NOTE:
        // Compute as meter.

        constexpr auto dD = Length::as(A_STEP, MeterUnit::mm);

        float droplet_diameter_mm = Length::as(A_MIN, MeterUnit::mm);

        float extinction_sum = 0.0F;

        for (int32_t a = 0; a < A_WIDTH; a++) {
            const auto droplet_diameter_m = Length::from(droplet_diameter_mm, MeterUnit::mm, MeterUnit::m);
#if 0
            const auto droplet_distrib = ComputeMarshallPalmerDropletSizeDistribution(
                droplet_diameter_m,
                intensity_rainfall_rate);

            const auto droplet_radius_mm = droplet_diameter_mm * 0.5F;
            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_mm * droplet_radius_mm;
#else
            // TODO
            // The following code should be standardized.
            constexpr auto N0 = 8000.0F * 10e3F;
            const auto lambda = 4.1F * 10e3F * aten::pow(intensity_rainfall_rate, -0.21F);
            const auto droplet_distrib = N0 * aten::exp(-lambda * droplet_diameter_m);
            const auto droplet_radius_m = droplet_diameter_m * 0.5F;
            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_m * droplet_radius_m;
#endif

            const auto extinction = EXTINCTION_EFFICIENT_IN_RAIN_VOLUME * droplet_cross_sectional_area * droplet_distrib;

            droplet_diameter_mm += dD;

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            float weight_i = a == 0 || a == A_WIDTH - 1 ? 0.5F : 1.0F;
            extinction_sum += extinction * weight_i * dD;
        }

        return extinction_sum;
    }

    inline AT_DEVICE_API aten::vec3 AdvanceRainVolumeIntegral(
        aten::sampler& sampler,
        const sky::AtmosphereParameters& atmosphere,
        const aten::sky::texture2d& transmittance_texture,
        const aten::sky::texture3d& droplet_radius_tex,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center, // [km]
        const aten::vec3& camera_pos,   // [km]
        const aten::vec3& view_dir,
        const aten::aabb& rain_volume,  // [km x km x km]
        const float intensity_rainfall_rate,    // [mm/h]
        const float extinction,
        const aten::sky::texture3d& airy_func_res_tex)
    {
        // If the view direction is the same as sun direction, the rainbow doesn't appear.
        const bool is_same_direction = dot(sun_direction, view_dir) > 0.0F;
        if (is_same_direction) {
            return aten::vec3(0.0F);
        }

        const auto sun_light_direction = -sun_direction;

        // TODO
        // そもそも、太陽が地球の下に隠れて見えないなどについては、
        // GetTransmittanceToSun など sky 側で対応済みなので、それを利用する.
        // ただ、その場合に太陽を点ではなく円盤としているので、太陽の扱いは円盤にすること.

        const float theta = aten::acos(dot(sun_light_direction, view_dir));
        if (theta < THETA_MIN || theta >= THETA_MAX) {
            // 主虹、副虹の範囲内に収まらないので、虹が見えない.
            return aten::vec3(0.0F);
        }

        aten::vec3 curr_point;

        float t0, t1;
        aten::tie(t0, t1) = aten::aabb::GetHitT(rain_volume, aten::ray(camera_pos, view_dir), AT_MATH_EPSILON, AT_MATH_INF);

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

        constexpr auto SAMPLE_COUNT = 100;

        const auto box_distance_along_with_view_dir = t1 - t0;

        // Not to out of the rain volume at the last step, multiply by 0.999999F.
        const auto dt = box_distance_along_with_view_dir / SAMPLE_COUNT * 0.999999F;

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

        float optical_length_in_rain_volume = 0.0F;

        aten::vec3 rainbow_radiance_sum{ 0.0F };

        const aten::vec3& move_dir = view_dir;

        for (size_t i = 0; i <= SAMPLE_COUNT; i++) {
            AT_ASSERT(rain_volume.isIn(curr_point));

#if 0
            const float droplet_radius = GetDropletRadiusFromPreComputeTexture(
                droplet_radius_tex,
                curr_point, rain_volume);
            const auto droplet_diameter = static_cast<float>(droplet_radius * 2.0F);
            float pdf = 1.0F;   // TODO.
#elif 0
            const auto u = sampler.nextSample();
            const auto droplet_radius = static_cast <float>(0.3_mm + (0.7_mm - 0.3_mm) * u);
            const auto droplet_diameter = static_cast<float>(droplet_radius * 2.0F);
            float pdf = static_cast<float>((0.7_mm - 0.3_mm).as(MeterUnit::mm));
#else
            const auto p_max = 1.0F - ComputeMarshallPalmerDropletSizeDistributionFactor(A_MAX * 2.0F, intensity_rainfall_rate);
            const auto p_min = 1.0F - ComputeMarshallPalmerDropletSizeDistributionFactor(A_MIN * 2.0F, intensity_rainfall_rate);

            auto u = sampler.nextSample();
            u = p_min + u * (p_max - p_min);

            auto droplet_diameter = GetDropletDiamterFromMarshallPalmerDropletSizeDistribution(u, intensity_rainfall_rate);
            droplet_diameter = Length::from(droplet_diameter, MeterUnit::mm, MeterUnit::m);
            const auto pdf = GetMarshallPalmerDropletSizeDistributionPDF(droplet_diameter, intensity_rainfall_rate);

            const auto droplet_radius = 0.5F * droplet_diameter;
#endif

            uvw.z = aten::saturate(((droplet_radius - A_MIN) / A_STEP + 0.5F) / A_WIDTH);

            const auto transmittance_to_sun = GetTransmittanceToSun(
                atmosphere,
                transmittance_texture,
                curr_point,
                sun_direction, earth_center);

            aten::vec3 rainbow_intensity;

            for (size_t i = 0; i < visible_wavelength.size(); i++) {
                const auto wavelength = visible_wavelength[i];
                uvw.y = aten::saturate(((wavelength - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F) / WAVELENGTH_WIDTH);
                rainbow_intensity[i] =  GetAiryFunctionValue(airy_func_res_tex, uvw);;
            }

#if 1
            constexpr float N0 = 8000.0F;
            const auto mp_lambda = ComputeMarshallPalmerDropletSizeDistributionLambda(intensity_rainfall_rate);

            const auto droplet_diameter_as_mm = Length::as(droplet_diameter, MeterUnit::mm);
            const auto droplet_radius_as_mm = droplet_diameter_as_mm * 0.5F;

            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_as_mm * droplet_radius_as_mm;
            const auto rain_density = N0 / mp_lambda * droplet_cross_sectional_area * (p_max - p_min);

            optical_length_in_rain_volume += dt;
            const auto transmittance_in_rain_volume = aten::exp(-extinction * optical_length_in_rain_volume);

            const auto rainbow_radiance{
                transmittance_in_rain_volume * transmittance_to_sun * solar_radiance * rainbow_intensity * rain_density
            };
#else
            auto rain_density = ComputeMarshallPalmerDropletSizeDistribution(
                droplet_diameter, intensity_rainfall_rate);

            const auto droplet_radius_as_mm = Length::as(droplet_radius, MeterUnit::mm);
            rain_density *= AT_MATH_PI * droplet_radius_as_mm * droplet_radius_as_mm;

            const auto rainbow_radiance{
                transmittance_to_sun* solar_radiance * rainbow_intensity * rain_density / pdf
            };
#endif
            curr_point += move_dir * dt;

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            float weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5F : 1.0F;
            rainbow_radiance_sum += rainbow_radiance * weight_i * dt;
        }

        return rainbow_radiance_sum;
    }
}
