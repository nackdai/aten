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
    inline AT_DEVICE_API float ComputeOpticalDepthBasedOnAabbCoveredHemiSphere(
        const sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const float r,
        const float mu)
    {
        AT_ASSERT(r <= (box.maxPos().y - box.minPos().y));

        // Take care from zenith to horizon.
        AT_ASSERT(mu >= 0.0 && mu <= 1.0);

        constexpr int32_t SAMPLE_COUNT = 100;

        const auto profile = atmosphere.mie_density;

        // Get diagonal length. It's the longest length and then it's enough for radius of aabb covered hemisphere
        const auto radius = box.getDiagonalLenght();

        const auto r_o = r - box.minPos().y + atmosphere.bottom_radius;

        const auto dx = radius / static_cast<float>(SAMPLE_COUNT);

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
            const float y_i = sky::GetProfileDensity(profile, r_i - atmosphere.bottom_radius);

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

        const auto Rt = atmosphere.top_radius;
        const auto Rg = atmosphere.bottom_radius;

        const float x_mu = sky::GetUnitRangeFromTextureCoord(uv.x, sky::TRANSMITTANCE_TEXTURE_WIDTH);
        const float x_r = sky::GetUnitRangeFromTextureCoord(uv.y, sky::TRANSMITTANCE_TEXTURE_HEIGHT);

        const float height = box.maxPos().y - box.minPos().y;

        r = height * x_r + box.minPos().y + atmosphere.bottom_radius;

        // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
        // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
        // from which we can recover mu:
        // 視線レイが真上を向いているとき.
        const float d_min = atmosphere.top_radius - r;

        // x から水平方向に向かってレイを飛ばしたときの大気までの距離.
        const float d_max = aten::sqrt(Rt * Rt - r * r);

        // x_mu は地球の中心Oをとある点Pを結ぶベクトルOPと視線レイvとの内積で [0, 1]
        // x_mu = 1 で視線レイvが地平線方向を向くので、d_max になる.
        // x_mu = 0 で視線レイvが上方向を向くので、d_min になる.
        const float d = d_min + x_mu * (d_max - d_min);

        mu = d == 0.0F
            ? 1.0F
            : -((r * r + d * d - Rt * Rt) / (2.0F * r * d));
        mu = aten::clamp(mu, 0.0F, 1.0F);
    }

    inline AT_DEVICE_API vec2 GetTransmittanceTextureUvFromRMu(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::aabb& box,
        const float r,
        const float mu)
    {
        // NOTE:
        // What we need to computed is just mu (dot). So, no need to care which length we use for the computation.
        // To simplify the computation, we use the earth radius and the top atmosphere radius.

        const auto Rt = atmosphere.top_radius;
        const auto Rg = atmosphere.bottom_radius;

        const auto height = box.maxPos().y - box.minPos().y;

        const float r_o = r - box.minPos().y - atmosphere.bottom_radius;

        AT_ASSERT(r_o <= height);

        // Take care from zenith to horizon.
        AT_ASSERT(mu >= 0.0 && mu <= 1.0);

        const float x_r = r_o / height;

        // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
        // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
        // 大気上端までの距離を計算.
        const float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);

        const float d_min = atmosphere.top_radius - r;

        // x から水平方向に向かってレイを飛ばしたときの大気までの距離.
        const float d_max = aten::sqrt(Rt * Rt - r * r);

        // d = d_min + x_mu * (d_max - d_min) という式で、x_mu から d を計算しているので、逆に、d から x_mu を計算するには、
        // x_mu = (d - d_min) / (d_max - d_min)
        const float x_mu = (d - d_min) / (d_max - d_min);

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
        AT_ASSERT(mu >= 0.0F && mu <= 1.0F);

        const auto height = box.maxPos().y - box.minPos().y;

        const float r_o = r - box.minPos().y - atmosphere.bottom_radius;

        AT_ASSERT(r_o <= height);

        const auto x_mu = mu;
        const auto x_r = r_o / height;

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
        auto z = point - earth_center;
        const auto r = length(z);

        static const aten::vec3 zenith{0.0F, 0.0F, 1.0F};

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
}
