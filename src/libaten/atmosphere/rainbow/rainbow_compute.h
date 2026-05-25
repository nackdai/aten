#pragma once

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

    inline AT_DEVICE_API float GetDropletDiameterFromMarshallPalmerDropletSizeDistribution(
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

        for (int32_t i = 0; i <= n_steps; i++) {
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

    inline AT_DEVICE_API aten::tuple<float, float, float> CieColorMatchingFunctionTableValue(
        const int32_t wavelength_nm)
    {
        if (wavelength_nm <= sky::LambdaMin || wavelength_nm >= sky::LambdaMax) {
            return aten::make_tuple(0.0F, 0.0F, 0.0F);
        }

        auto u = (wavelength_nm - sky::LambdaMin) / 5.0F;
        const auto row = static_cast<int32_t>(aten::floor(u));

#if __CUDACC__
        // TODO
        // まずは、ホスト側での動作を試す.
        sky::CIE_2_DEG_COLOR_MATCHING_FUNCTIONS_ELEMENT e0;
        sky::CIE_2_DEG_COLOR_MATCHING_FUNCTIONS_ELEMENT e1;
#else
        const auto& e0 = sky::CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row];
        const auto& e1 = sky::CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row + 1];
#endif

        u -= row;

        return aten::make_tuple(
            aten::lerp(e0.x, e1.x, u),
            aten::lerp(e0.y, e1.y, u),
            aten::lerp(e0.z, e1.z, u));
    }

    inline AT_DEVICE_API aten::vec3 SpectrumSampleToLinearSrgb(
        const int32_t wavelength_nm,
        const float spectral_value,
        const float dlambda_nm)
    {
        float x_bar, y_bar, z_bar;
        aten::tie(x_bar, y_bar, z_bar) = CieColorMatchingFunctionTableValue(wavelength_nm);

        const float x = spectral_value * x_bar;
        const float y = spectral_value * y_bar;
        const float z = spectral_value * z_bar;

        return sky::MAX_LUMINOUS_EFFICACY * dlambda_nm * aten::vec3{
            sky::XYZ_TO_SRGB[0] * x + sky::XYZ_TO_SRGB[1] * y + sky::XYZ_TO_SRGB[2] * z,
            sky::XYZ_TO_SRGB[3] * x + sky::XYZ_TO_SRGB[4] * y + sky::XYZ_TO_SRGB[5] * z,
            sky::XYZ_TO_SRGB[6] * x + sky::XYZ_TO_SRGB[7] * y + sky::XYZ_TO_SRGB[8] * z,
        };
    }

    inline AT_DEVICE_API aten::vec3 ComputeSpectralRgbPhaseValue(
        const int32_t theta_idx,
        const int32_t radius_idx)
    {
        aten::vec3 rgb{ 0.0F };

        constexpr int32_t LAMBDA_STEP_NM = 10;
        constexpr float dlambda_nm = static_cast<float>(LAMBDA_STEP_NM);

        for (int32_t lambda_nm = sky::LambdaMin; lambda_nm <= sky::LambdaMax; lambda_nm += LAMBDA_STEP_NM) {
            const auto lambda = Length::from(static_cast<float>(lambda_nm), MeterUnit::nm, MeterUnit::m);
            const int32_t wavelength_idx = aten::clamp(
                static_cast<int32_t>((lambda - WAVELENGTH_MIN) / WAVELENGTH_STEP + 0.5F),
                0,
                WAVELENGTH_WIDTH - 1
            );

            const float airy = ComputeAiryFunction(theta_idx, wavelength_idx, radius_idx);

            rgb += SpectrumSampleToLinearSrgb(
                lambda_nm,
                airy,
                dlambda_nm);
        }

        return rgb;
    }
}
