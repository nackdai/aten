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
        constexpr int32_t SAMPLE_COUNT = 100;

        const auto profile = atmosphere.mie_density;

        // Get diagonal length. It's the longest length and then it's enough for radius of aabb covered hemisphere
        const auto R = box.getDiagonalLenght() + 1.0F;

        AT_ASSERT(
            r >= atmosphere.bottom_radius
            && (r - atmosphere.bottom_radius) <= (box.maxPos().y - box.minPos().y));
        const auto r_o = r - box.minPos().y - atmosphere.bottom_radius + 1.0F;

        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        const auto dx = R / static_cast<float>(SAMPLE_COUNT);

        float result = 0.0F;

        for (int32_t i = 0; i <= SAMPLE_COUNT; i++)
        {
            const float d_i = i * dx;

            // Distance between the current sample point and the planet center.
            // d_i での高度を計算.
            // |x + ts|^2 = r_i^2
            // => |x|^2 + 2tx・s + t^2|s|^2 = r_i^2
            // => r^2 + 2rμ + t^2 = r_i^2 (|x|=r, x・s=μ, |s|=1)
            // tについて解くと、t = -rμ ± sqrt(r^2(μ^2-1)+r_i^2) となるが、t = d_i となるため、
            // r_i^2 = d_i^2 + 2rμd_i + r^2 となる.
            const float r_i = aten::sqrt(d_i * d_i + 2.0F * r_o * mu * d_i + r_o * r_o);

            // atmosphere.bottom_radius : R_ground

            // Number density at the current sample point (divided by the number density
            // at the bottom of the atmosphere, yielding a dimensionless number).
            // Rayleigh散乱層、Mie散乱層の場合だと、exp(-h/H) を計算する.
            const float y_i = sky::GetProfileDensity(profile, r_i - 1.0F);

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            float weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5F : 1.0F;
            result += y_i * weight_i * dx;
        }

        return result;
    }

    inline AT_DEVICE_API void GetRMuFromTransmittanceTextureUv(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const aten::vec2& uv,
        float& r,
        float& mu)
    {
        // NOTE:
        // What we need to computed is just mu (dot). So, no need to care which length we use for the computation.
        // To simplify the computation, we use the earth radius and the top atmosphere radius.

        AT_ASSERT(uv.x >= 0.0 && uv.x <= 1.0);
        AT_ASSERT(uv.y >= 0.0 && uv.y <= 1.0);

        const float x_r = sky::GetUnitRangeFromTextureCoord(uv.y, sky::TRANSMITTANCE_TEXTURE_HEIGHT);

        const float height = box.maxPos().y - box.minPos().y;

        r = height * x_r + 1.0F;

        // Get diagonal length. It's the longest length and then it's enough for radius of aabb covered hemisphere
        const auto R = box.getDiagonalLenght() + 1.0F;

        float x_mu = sky::GetUnitRangeFromTextureCoord(uv.x, sky::TRANSMITTANCE_TEXTURE_WIDTH);

        // [0, 1] -> [-1, 1]
        x_mu = 2.0F * x_mu - 1.0F;

        float d = 0.0F;

        // x_mu = 1 で視線レイvが"上"方向を向く => mu = 1.
        // x_mu = 0 で視線レイvが"水平"方向を向く => mu = 0.
        // x_mu = -1 で視線レイvが"下"方向を向く => mu = -1.

        if (x_mu >= 0.0F) {
            // 視線レイが真上を向いているとき (x_mu = 1).
            const float d_min = R - r;

            // 視線レイが水平を向いているとき (x_mu = 0).
            const float d_max = aten::sqrt(R * R - r * r);

            d = d_min + x_mu * (d_max - d_min);
        }
        else {
            // 視線レイが水平を向いているとき (x_mu = 0).
            const float d_min = aten::sqrt(R * R - r * r);

            // 視線レイが真下を向いているとき (x_mu = 1).
            const float d_max = R + r;

            d = d_min - x_mu * (d_max - d_min);
        }

        // 余弦定理から、計算.
        // https://gemini.google.com/share/91af735989be
        mu = d == 0.0F
            ? 1.0F
            : -(r * r + d * d - R * R) / (2.0F * r * d);

        mu = aten::clamp(mu, -1.0F, 1.0F);

        r = height * x_r + box.minPos().y + atmosphere.bottom_radius;
    }

    inline AT_DEVICE_API float DistanceToSphereBoundary(
        const float radius,
        const float r,
        const float mu)  // 太陽方向の余弦.
    {
        AT_ASSERT(r <= radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // |x + ts|^2 = R^2 から、tを求めたときの判別式.
        const float discriminant = r * r * (mu * mu - 1.0F) + radius * radius;

        // SafeSqrtで、discriminantが負のときは0を返す.

        // |x + ts|^2 = R^2 から、tを求めると、解は２つあるが、-r * mu - SafeSqrt(discriminant) < 0 となるので、
        // -r * mu + SafeSqrt(discriminant) の方が、距離dとして正しい値になる.
        return aten::max(-r * mu + sky::safe_sqrt(discriminant), 0.0F);
    }

    inline AT_DEVICE_API vec2 GetTransmittanceTextureUvFromRMu(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const float r,
        const float mu)
    {
        // Get diagonal length. It's the longest length and then it's enough for radius of aabb covered hemisphere
        const auto R = box.getDiagonalLenght() + 1.0F;

        const auto height = box.maxPos().y - box.minPos().y;

        float r_o = r - box.minPos().y - atmosphere.bottom_radius;

        AT_ASSERT(r_o <= height);

        const float x_r = r_o / height;
        r_o += 1.0F;

        // Take care from zenith to horizon.
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
        // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
        // 大気上端までの距離を計算.
        const float d = DistanceToSphereBoundary(R, r_o, mu);

        float x_mu = 0.0F;

        // x_mu = 1 で視線レイvが"上"方向を向く => mu = 1.
        // x_mu = 0 で視線レイvが"水平"方向を向く => mu = 0.
        // x_mu = -1 で視線レイvが"下"方向を向く => mu = -1.

        if (mu >= 0.0F) {
            // 視線レイが真上を向いているとき (x_mu = 1).
            const float d_min = R - r_o;

            // 視線レイが水平を向いているとき (x_mu = 0).
            const float d_max = aten::sqrt(R * R - r_o * r_o);

            x_mu = (d - d_min) / (d_max - d_min);
        }
        else {
            // 視線レイが水平を向いているとき (x_mu = 0).
            const float d_min = aten::sqrt(R * R - r_o * r_o);

            // 視線レイが真下を向いているとき (x_mu = 1).
            const float d_max = R + r_o;

            x_mu = -(d - d_min) / (d_max - d_min);
        }

        // [-1, 1] -> [0, 1]
        x_mu = 0.5F * x_mu + 0.5F;
        x_mu = aten::saturate(x_mu);

        // x_mu と x_r をテクスチャのUV座標に変換する.
        return vec2(
            sky::GetTextureCoordFromUnitRange(x_mu, sky::TRANSMITTANCE_TEXTURE_WIDTH),
            sky::GetTextureCoordFromUnitRange(x_r, sky::TRANSMITTANCE_TEXTURE_HEIGHT));
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceInRainVolumeByRMu(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const sky::texture2d& transmission_rain_tex,
        const float r,
        const float mu)
    {
        const auto height = box.maxPos().y - box.minPos().y;

        const float r_o = r - box.minPos().y - 1.0F; //atmosphere.bottom_radius;

        AT_ASSERT(r_o <= height);

        const auto x_r = r_o / height;

        AT_ASSERT(mu >= -1.0F && mu <= 1.0F);

        // [-1, 1] -> [0, 1]
        const auto x_mu = 0.5F * mu + 0.5F;

        aten::vec2 uv{
            sky::GetTextureCoordFromUnitRange(x_mu, sky::TRANSMITTANCE_TEXTURE_WIDTH),
            sky::GetTextureCoordFromUnitRange(x_r, sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        };

        const auto transmittance = sky::SampleTexture2D(transmission_rain_tex, uv);
        return transmittance;
    }

    inline AT_DEVICE_API aten::tuple<float, float> ComputeRMu(
        const sky::AtmosphereParameters& atmosphere,
        const aten::vec3& earth_center,
        const aten::vec3& point,
        const aten::vec3& view_dir)
    {
        // TODO
        // ここは、earth center から考えるのではなく、ダミーで設定した 1.0 の半径の球の中心から考えるべきかもしれない.
        auto z = point - earth_center;
        const auto r = length(z);

        // TODO
        // static in CUDA
        const aten::vec3 zenith{0.0F, 0.0F, 1.0F};

        const auto mu = dot(zenith, view_dir);

        return aten::make_tuple(r, mu);
    }

    inline AT_DEVICE_API aten::vec3 GetTransmittanceInRainVolumeBetweenTwoPoints(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const aten::vec3& earth_center,
        const sky::texture2d& transmission_rain_tex,
        const aten::vec3& point_1,
        const aten::vec3& point_2)
    {
        const aten::vec3 view_dir{ normalize(point_2 - point_1) };

        float r_1, mu_1;
        aten::tie(r_1, mu_1) = ComputeRMu(atmosphere, earth_center, point_1, view_dir);
        const auto transmittance_1 = GetTransmittanceInRainVolumeByRMu(atmosphere, box, transmission_rain_tex, r_1, mu_1);

        float r_2, mu_2;
        aten::tie(r_2, mu_2) = ComputeRMu(atmosphere, earth_center, point_2, view_dir);
        const auto transmittance_2 = GetTransmittanceInRainVolumeByRMu(atmosphere, box, transmission_rain_tex, r_2, mu_2);

        // transmittance は視点をx、大気上端の交点を i とすると、 exp(-∫_x^i).
        // また、途中の点を y とすると、 exp(-∫_x^i) = exp{-(∫_x^y + ∫_y^i}) = exp(-∫_x^y) * exp(-∫_y^i) となる.
        // つまり、exp(-∫_x^y) = exp(-∫_x^i) / exp(-∫_y^i) となるので、
        // transmittance(x,y) = transmittance(x,i) / transmittance(y,i) となる.
        // x ----- y ----- i

        const auto transmittance = transmittance_1 / transmittance_2;
        return transmittance;
    }

    // TODO
    inline AT_DEVICE_API float ComputeExtinctionInRain(
        const float intensity_rainfall_rate)    // [mm/h]
    {
        // NOTE:
        // Compute as km.

        constexpr auto dD = Length::as(A_STEP, MeterUnit::mm);

        float droplet_diameter_mm = Length::as(A_MIN, MeterUnit::mm);

        float extinction = 0.0F;

        for (int32_t a = 0; a < A_WIDTH; a++) {
            const auto droplet_diameter_m = Length::from(droplet_diameter_mm, MeterUnit::mm, MeterUnit::km);
#if 0
            const auto droplet_distrib = ComputeMarshallPalmerDropletSizeDistribution(
                droplet_diameter_m,
                intensity_rainfall_rate);

            const auto droplet_radius_mm = droplet_diameter_mm * 0.5F;
            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_mm * droplet_radius_mm;
#else
            // TODO
            // The following code should be standardized.
            constexpr auto N0 = 8000.0F * 10e3F * 10e8F;    // [m^-3mm^-1] -> [m^-4] -> [km^-4]
            const auto lambda = 4.1F * 10e3F * 10e2F * aten::pow(intensity_rainfall_rate, -0.21F);
            const auto droplet_distrib = N0 * aten::exp(-lambda * droplet_diameter_m);
            const auto droplet_radius_m = droplet_diameter_m * 0.5F;
            const auto droplet_cross_sectional_area = AT_MATH_PI * droplet_radius_m * droplet_radius_m;
#endif

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
}
