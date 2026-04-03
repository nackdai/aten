#pragma once

#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_types.h"

#include "math/vec2.h"

// NOTE
// 計算単位
// - meter

namespace aten::sky {
    inline AT_DEVICE_API aten::vec3 SampleTexture2D(
        const texture2d& tex,
        const aten::vec2& uv)
    {
#ifdef __CUDACC__
        AT_ASSERT(tex.texture > 0);
        auto tmp = tex2D<float4>(tex.texture, uv.x, uv.y);
        return { tmp.x, tmp.y, tmp.z };
#else
        return tex.AtWithBilinear(uv.x, uv.y);
#endif
    }

    inline AT_DEVICE_API void WriteTexture2D(
        texture2d& tex,
        const aten::vec3& value,
        int32_t x, int32_t y)
    {
#ifdef __CUDACC__
        AT_ASSERT(tex.surface > 0);
        surf2Dwrite(
            make_float4(value.x, value.y, value.z, 1.0F),
            tex.surface,
            x * sizeof(float4), y,
            cudaBoundaryModeTrap);
#else
        tex.PutByXYcoord(x, y, value);
#endif
    }

    inline AT_DEVICE_API aten::vec3 SampleTexture3D(
        const texture3d& tex,
        const aten::vec3& uvw)
    {
#ifdef __CUDACC__
        AT_ASSERT(tex.texture > 0);
        auto tmp = tex3D<float4>(tex.texture, uvw.x, uvw.y, uvw.z);
        return { tmp.x, tmp.y, tmp.z };
#else
        return tex.AtWithTrilinear(uvw.x, uvw.y, uvw.z);
#endif
    }

    inline AT_DEVICE_API void WriteTexture3D(
        texture3d& tex,
        const aten::vec3& value,
        int32_t x, int32_t y, int32_t z)
    {
#ifdef __CUDACC__
        AT_ASSERT(tex.surface > 0);
        surf3Dwrite(
            make_float4(value.x, value.y, value.z, 1.0F),
            tex.surface,
            x * sizeof(float4), y, z,
            cudaBoundaryModeTrap);
#else
        tex.SetByXYZ(value, x, y, z);
#endif
    }

    inline AT_DEVICE_API float safe_sqrt(const float x)
    {
        return x <= 0.0F ? 0.0F : aten::sqrt(x);
    }

    // 大気の上端境界までの距離を計算する.
    inline AT_DEVICE_API float DistanceToTopAtmosphereBoundary(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu)  // 太陽方向の余弦.
    {
        AT_ASSERT(r <= atmosphere.top_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // atmosphere.top_radius : R_t

        // |x + ts|^2 = R_t^2 から、tを求めたときの判別式.
        const float discriminant = r * r * (mu * mu - 1.0F) +
            atmosphere.top_radius * atmosphere.top_radius;

        // SafeSqrtで、discriminantが負のときは0を返す.

        // |x + ts|^2 = R_t^2 から、tを求めると、解は２つあるが、-r * mu - SafeSqrt(discriminant) < 0 となるので、
        // -r * mu + SafeSqrt(discriminant) の方が、距離dとして正しい値になる.
        return aten::max(-r * mu + safe_sqrt(discriminant), 0.0F);
    }

    // 地球との交点までの距離を計算する.
    inline AT_DEVICE_API float DistanceToBottomAtmosphereBoundary(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // |x + ts|^2 = R_g^2 から、tを求めたときの判別式.
        const float discriminant = r * r * (mu * mu - 1.0F) +
            atmosphere.bottom_radius * atmosphere.bottom_radius;

        // |x + ts|^2 = R_g^2 から、tを求めると、解は２つあり、μ < 0 （視線が地面方向のため）であるために二つとも 正の値になる.
        // 視線レイが地面と二点で交差する場合に、地球の奥側は見えないので、手前だけ考えればよく、二つの解のうち小さい方を選べばいい.
        // 二つの解は、-r * mu ± SafeSqrt(discriminant) なので、-r * mu - SafeSqrt(discriminant) の方が小さい方になる.
        return aten::max(-r * mu - safe_sqrt(discriminant), 0.0F);
    }

    inline AT_DEVICE_API float DistanceToNearestAtmosphereBoundary(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu,
        const bool ray_r_mu_intersects_ground)
    {
        if (ray_r_mu_intersects_ground) {
            return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
        }
        else {
            return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
        }
    }

    // 論文の式(2)
    inline AT_DEVICE_API float RayleighPhaseFunction(const float nu)
    {
        constexpr float k = 3.0F / (16.0F * AT_MATH_PI);
        return k * (1.0F + nu * nu);
    }

    // 論文の式(4)
    inline AT_DEVICE_API float MiePhaseFunction(const float g, const float nu) {
        const float k = 3.0F / (8.0F * AT_MATH_PI) * (1.0F - g * g) / (2.0F + g * g);
        return k * (1.0F + nu * nu) / aten::pow(1.0F + g * g - 2.0F * g * nu, 1.5F);
    }

    // 視線レイが地面と交差するかどうかを計算する.
    inline AT_DEVICE_API bool RayIntersectsGround(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // そもそも、視線が地面方向（下方向）を向いていない場合は、地面と交差しない.
        // そのため、mu < 0.0 であることが必要条件になる.

        // |x + ts|^2 = R_g^2 から、tを求めたときの判別式:
        //   D = r * r * (mu * mu - 1.0) + atmosphere.bottom_radius * atmosphere.bottom_radius
        // このDが正（>=0）のとき、視線レイは地面と交差することになる.

        return mu < 0.0F
            && r * r * (mu * mu - 1.0F) + atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0F;
    }

    inline float InterpolateFactor(
        const std::vector<float>& interp_factors,
        const std::vector<float>& base_values,
        const float v)
    {
        AT_ASSERT(interp_factors.size() == base_values.size());

        if (v < base_values[0]) {
            return interp_factors[0];
        }

        for (size_t i = 0; i < base_values.size() - 1; ++i) {
            if (v < base_values[i + 1]) {
                const auto u = (v - base_values[i]) / (base_values[i + 1] - base_values[i]);
                return interp_factors[i] * (1.0F - u) + interp_factors[i + 1] * u;
            }
        }

        return interp_factors[interp_factors.size() - 1];
    }

    // 波長ごとのfactorsの値をRGB波長の値に応じた値になるように補間する.
    inline aten::vec3 InterpolateFactorByRGBLambda(
        const std::vector<float> factors,
        const std::vector<float> wavelengths,
        const aten::vec3& rgb_lambda)
    {
        const auto r = InterpolateFactor(factors, wavelengths, rgb_lambda.r);
        const auto g = InterpolateFactor(factors, wavelengths, rgb_lambda.g);
        const auto b = InterpolateFactor(factors, wavelengths, rgb_lambda.b);
        return aten::vec3{ r, g, b };
    }
}
