#pragma once

#include "atmosphere/rainbow/rainbow_constants.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_render.h"
#include "atmosphere/sky/unit_quantity.h"

#include "math/aabb.h"
#include "image/texture_3d.h"

namespace aten::rainbow {
    // マーシャル・パルマー粒径分布を計算.
    // intensity_rainfall_rate[mm/h]
    // droplet_diameter[m]
    inline float ComputeMarshallPalmerDropletSizeDistribution(
        const float droplet_diameter,
        const float intensity_rainfall_rate)
    {
        // https://en.wikipedia.org/wiki/Raindrop_size_distribution
        // https://www.atmos.albany.edu/facstaff/rfovell/ATM562/marshall-palmer-1948.pdf

        // 論文だと、N0 = 0.008[cm^-4] で計算している.
        // その場合は、D [cm] で計算する必要がある.

        // [cm^-4]
        constexpr float N0 = 0.008F;

        // m -> cm.
        const auto D = droplet_diameter * 100.0F;

        const auto lambda = 41.0F * aten::pow(intensity_rainfall_rate, -0.21F);

        const auto ND = N0 * aten::exp(-lambda * D);

        return ND;
    }

    // 水の屈折率(20°C)を計算.
    // wavelength[m] e.g. 660e-9[m]
    inline float ComputeWaterRefractiveIndex(const float wavelength)
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

    inline float ComputeH(const float n)
    {
        // n : 屈折率
        const auto n2 = aten::pow(n, 2);
        const auto h = 9.0F / (4.0F * (n2 - 1.0F)) * aten::sqrt((4.0F - n2) / (n2 - 1.0F));
        return h;
    }

    inline float ComputeThetaMax(const float n)
    {
        // n : 屈折率
        const auto n2 = aten::pow(n, 2);
        const auto theta_max = 4.0F * aten::asin(aten::sqrt((4.0F - n2) / (3.0F * n2))) - 2.0F * aten::asin(aten::sqrt((4.0F - n2) / 3.0F));
        return theta_max;
    }

    inline float ComputeZ(
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

    inline float ComputeM(
        const float k,
        const float wavelength, // [m]
        const float a,  // droplet diamter. [m]
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

    inline float ComputeAiryFunctionByIntegral(const float z)
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

    inline void PreComputeAiryFunction(aten::texture3d& airy_func_tex)
    {
        for (int32_t z = 0; z < A_WIDTH; z++) {
            // Droplet diameter. Unit is [m].
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

                    airy_func_tex.SetByXYZ(vec4(M * M * f_z * f_z), x, y, z);
                }
            }
        }
    }

    inline float ComputeAiryFunction(
        int32_t x, int32_t y, int32_t z)
    {
        // Droplet diameter. Unit is [m].
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

    inline float GetAiryFunctionValue(
        const aten::texture3d& airy_func_tex,
        const aten::vec3& uvw)
    {
        AT_ASSERT(uvw.x >= 0.0F && uvw.x <= 1.0F);
        AT_ASSERT(uvw.y >= 0.0F && uvw.y <= 1.0F);
        AT_ASSERT(uvw.z >= 0.0F && uvw.z <= 1.0F);

        // NOTE:
        // the content value is the same in vec3.
        return sky::SampleTexture3D(airy_func_tex, uvw).x;
    }

    inline float GetAiryFunctionValue(
        const aten::texture3d& airy_func_tex,
        const float wavelength, // [m]
        const float a,  // droplet_diameter [m]
        const float theta)
    {
        aten::vec3 uvw{
            ((theta - THETA_MIN) / THETA_STEP + 0.5F) / THETA_WIDTH,
            ((wavelength - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F) / WAVELENGTH_WIDTH,
            ((a - A_MIN) / A_STEP + 0.5F) / A_WIDTH
        };

        return GetAiryFunctionValue(airy_func_tex, uvw);
    }

    inline aten::vec3 AdvanceRainVolumeIntegral(
        const sky::AtmosphereParameters& atmosphere,
        const aten::sky::texture2d& transmittance_texture,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center, // [km]
        const aten::vec3& camera_pos,   // [km]
        const aten::vec3& view_dir,
        const aten::aabb& rain_volume,  // [km x km x km]
        const float droplet_diameter,   // [m]
        const float intensity_rainfall_rate,    // [mm/h]
        const aten::texture3d& airy_func_res_tex)
    {
        // If the view direction is the same as sun direction, the rainbow doesn't appear.
        const bool is_same_direction = dot(sun_direction, view_dir) > 0.0F;
        if (is_same_direction) {
            return aten::vec3(0.0F);
        }

        const auto sun_light_direction = -sun_direction;

        constexpr Length MOVE_STEP = 0.2_km;
        const float dt = MOVE_STEP.as(MeterUnit::km);

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
        float hit_t = 0.0F;

        if (rain_volume.isIn(camera_pos)) {
            curr_point = camera_pos;
        }
        // TODO: t_near
        else if (rain_volume.hit(aten::ray(camera_pos, view_dir), AT_MATH_EPSILON, AT_MATH_INF, &hit_t)) {
            curr_point = camera_pos + hit_t * view_dir;
        }
        else {
            return aten::vec3(0.0F);
        }

        aten::vec3 uvw{
            aten::saturate(((theta - THETA_MIN) / THETA_STEP + 0.5F) / THETA_WIDTH),
            0.0F,   // compute while the integral calculation.
            aten::saturate(((droplet_diameter - A_MIN) / A_STEP + 0.5F) / A_WIDTH),
        };

        const auto solar_radiance{ sky::GetSolarRadiance(atmosphere) };

        // [nm] -> [m].
        constexpr std::array visible_wavelength = {
            sky::LambdaR * 1e-9F,
            sky::LambdaG * 1e-9F,
            sky::LambdaB * 1e-9F,
        };

        aten::vec3 rainbow_radiance{ 0.0F };

        const aten::vec3& move_dir = view_dir;

        for (;;) {
            curr_point += move_dir * dt;
            if (!rain_volume.isIn(curr_point)) {
                break;
            }

            auto z = curr_point - earth_center;
            const auto r = length(z);

            z /= r;
            const auto mu_s = dot(z, sun_direction);

            const auto transmittance = sky::transmittance::GetTransmittanceToSun(
                atmosphere,
                transmittance_texture,
                r, mu_s);

            aten::vec3 rainbow_intensity;

            for (size_t i = 0; i < visible_wavelength.size(); i++) {
                const auto wavelength = visible_wavelength[i];
                uvw.y = aten::saturate(((wavelength - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F) / WAVELENGTH_WIDTH);
                rainbow_intensity[i] =  GetAiryFunctionValue(airy_func_res_tex, uvw);;
            }

            rainbow_radiance += transmittance * solar_radiance * rainbow_intensity;
        }

        const auto rain_density = ComputeMarshallPalmerDropletSizeDistribution(
            droplet_diameter, intensity_rainfall_rate);
        rainbow_radiance *= rain_density;

        rainbow_radiance *= dt;

        return rainbow_radiance;
    }
}
