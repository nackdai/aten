#pragma once

#include "defs.h"

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_params.h"

#include "math/vec2.h"

namespace aten::sky {
    /*
        For this we need a mapping between the function parameters $(r, \mu)$ and the
        texture coordinates $(u, v)$, and vice - versa, because these parameters do not
        have the same units and range of values.And even if it was the case, storing a
        function $f$ from the $[0, 1]$ interval in a texture of size $n$ would sample the
        function at $0.5 / n$, $1.5 / n$, ... $(n - 0.5) / n$, because texture samples are at
        the center of texels.Therefore, this texture would only give us extrapolated
        function values at the domain boundaries($0$ and $1$).To avoid this we need
        to store $f(0)$ at the center of texel 0 and $f(1)$ at the center of texel
        $n - 1$.This can be done with the following mapping from values $x$ in $[0, 1]$ to
        texture coordinates $u$ in $[0.5 / n, 1 - 0.5 / n]$ - and its inverse :
    */

    float GetTextureCoordFromUnitRange(float x, float texture_size)
    {
        return 0.5F / texture_size + x * (1.0F - 1.0F / texture_size);
    }

    float GetUnitRangeFromTextureCoord(float u, float texture_size)
    {
        return (u - 0.5F / texture_size) / (1.0F - 1.0F / texture_size);
    }

    // For Transmittance ============================================

    inline void GetRMuFromTransmittanceByTextureUv(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::vec2& uv,
        float& r,
        float& mu)
    {
        AT_ASSERT(uv.x >= 0.0 && uv.x <= 1.0);
        AT_ASSERT(uv.y >= 0.0 && uv.y <= 1.0);

        float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
        float x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);

        // NOTE
        // ここでは、地面とは交差しない場合のみを考える.
        // そのため、視線レイの向きは地球に接する方向（地平線に接する方向）が下端となる.

        // Distance to top atmosphere boundary for a horizontal ray at ground level.
        // 論文中の 4. Precomputations の H = (Rt^2 - Rg^2)^1/2 に相当する値.
        // 地球のグラウンドレベルからの太陽が地平線にあるときの距離.
        const float H = aten::sqrt(atmosphere.top_radius * atmosphere.top_radius -
            atmosphere.bottom_radius * atmosphere.bottom_radius);

        // Distance to the horizon, from which we can compute r:

        // ある点Pからの地平線までの距離.
        // ρの最大値は点Pが大気の上端にあり、そこから地平線までの距離になるので、sqrt(Rt^2 - Rg^2) = H になる.
        // そして、ρの値はそこから0まで変化するので、x_r を0から1の値として、rho = H * x_r としている.
        const float rho = H * x_r;

        // 地球の中心Oから点Pまでの距離.
        r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

        // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
        // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
        // from which we can recover mu:
        // 視線レイが真上を向いているとき.
        const float d_min = atmosphere.top_radius - r;

        // 視線レイが地平線に接したとき.
        const float d_max = rho + H;

        // x_mu は地球の中心Oをとある点Pを結ぶベクトルOPと視線レイvとの内積で [0, 1]
        // x_mu = 1 で視線レイvが地平線方向を向くので、d_max になる.
        // x_mu = 0 で視線レイvが上方向を向くので、d_min になる.
        const float d = d_min + x_mu * (d_max - d_min);

        // この関数では、視線レイが大気に交差するので、
        // GetRMuMuSNuFromScatteringTextureUvzw での mu の計算と同様に、余弦定理で計算.
        mu = d == 0.0
            ? 1.0F
            : (H * H - rho * rho - d * d) / (2.0 * r * d);
        mu = aten::clamp(mu, -1.0F, 1.0F);
    }

    vec2 GetTransmittanceTextureUvFromRMu(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // Distance to top atmosphere boundary for a horizontal ray at ground level.
        // 論文中の 4. Precomputations の H = (Rt^2 - Rg^2)^1/2 に相当する値.
        // 地球のグラウンドレベルからの太陽が地平線にあるときの距離.
        const float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
            atmosphere.bottom_radius * atmosphere.bottom_radius);

        // Distance to the horizon.
        // ある点からの地平線までの距離.
        const float rho = safe_sqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);

        // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
        // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
        // 大気上端までの距離を計算.
        const float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);

        const float d_min = atmosphere.top_radius - r;
        const float d_max = rho + H;

        // GetRMuFromTransmittanceTextureUv で
        // d = d_min + x_mu * (d_max - d_min) という式で、x_mu から d を計算しているので、逆に、d から x_mu を計算するには、
        // x_mu = (d - d_min) / (d_max - d_min)
        const float x_mu = (d - d_min) / (d_max - d_min);

        // GetRMuFromTransmittanceTextureUv で
        // rho = H * x_r という式で、x_r から rho を計算しているので、逆に、rho から x_r を計算するには、
        // x_r = rho / H
        const float x_r = rho / H;

        // x_mu と x_r をテクスチャのUV座標に変換する.
        return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
            GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
    }

    // For Irradiance ============================================

    void GetRMuSFromIrradianceTextureUv(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::vec2& uv,
        float& r,
        float& mu_s)
    {
        AT_ASSERT(uv.x >= 0.0 && uv.x <= 1.0);
        AT_ASSERT(uv.y >= 0.0 && uv.y <= 1.0);

        const float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
        const float x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);

        // [0,1] -> [Rg, Rt] に変換.
        r = atmosphere.bottom_radius +
            x_r * (atmosphere.top_radius - atmosphere.bottom_radius);

        // [0,1] -> [-1, 1] に変換.
        mu_s = aten::clamp(2.0F * x_mu_s - 1.0F, -1.0F, 1.0F);
    }

    aten::vec2 GetIrradianceTextureUvFromRMuS(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu_s)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);

        const float x_r = (r - atmosphere.bottom_radius) / (atmosphere.top_radius - atmosphere.bottom_radius);
        const float x_mu_s = mu_s * 0.5F + 0.5F;

        return aten::vec2(
            GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
            GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
    }

    // For scattering ==============================================

    namespace _detail {
        void GetRMuMuSNuFromScatteringTextureByTexCoord(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::vec4& uvwz,
            float& r,
            float& mu,
            float& mu_s,
            float& nu,
            bool& ray_r_mu_intersects_ground)
        {
            AT_ASSERT(uvwz.x >= 0.0 && uvwz.x <= 1.0);
            AT_ASSERT(uvwz.y >= 0.0 && uvwz.y <= 1.0);
            AT_ASSERT(uvwz.z >= 0.0 && uvwz.z <= 1.0);
            AT_ASSERT(uvwz.w >= 0.0 && uvwz.w <= 1.0);

            // Distance to top atmosphere boundary for a horizontal ray at ground level.
            // 論文中の 4. Precomputations の H = (Rt^2 - Rg^2)^1/2 に相当する値.
            // 地球のグラウンドレベルからの太陽が地平線にあるときの距離.
            const float H = aten::sqrt(atmosphere.top_radius * atmosphere.top_radius -
                atmosphere.bottom_radius * atmosphere.bottom_radius);

            // 論文中の 4. Precomputations で、u_r = ρ/H.
            // uvwz.w には r ではなく、u_r が格納されているため、H を掛けて距離 ρ を求める.
            // Distance to the horizon.
            const float rho =
                H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);

            // r は、論文中の 4. Precomputations で、ρ = sqrt(r^2 - Rg^2) から計算.
            r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

            if (uvwz.z < 0.5) {
                // 視線レイが地面と交差する場合.
                // Distance to the ground for the ray (r,mu), and its minimum and maximum
                // values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
                // we can recover mu:

                // 視点 x から真下を向いたとき.
                const float d_min = r - atmosphere.bottom_radius;

                // 視点 x から地球に接する点までの距離は ρ で、それが地球に接するときの最大距離になる.
                // それを超えると、視線レイが地面と交差しなくなり、大気の上端と交差するようになる.
                const float d_max = rho;

                // uvwz.z に格納されているのは、u_μ. そして、地面と交差する場合の、
                // 値の範囲は [0, 0.5] で、d_min に対応するのが 0.5 で、d_max に対応するのが 0.0 になるように、GetUnitRangeFromTextureCoord の引数を調整している.
                // u_μ = 1/2 - d/2ρ で、u_μ = 1/2 のときは d = 0 （最小距離）で、u_μ = 0 のときは d = ρ (最大距離) になる.
                const float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                    1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);

                // d_min = 0 のときだけ d == 0 になり、それは、視点 x が地面に接しているときで、そのときを mu = -1 に対応させる.
                // 余弦定理から、計算.
                // https://gemini.google.com/share/91af735989be
                mu = d == 0.0F
                    ? -1.0F
                    : aten::clamp(-(rho * rho + d * d) / (2.0F * r * d), -1.0F, 1.0F);
                ray_r_mu_intersects_ground = true;
            }
            else {
                // 視線レイが大気の上端と交差する場合.
                // Distance to the top atmosphere boundary for the ray (r,mu), and its
                // minimum and maximum values over all mu - obtained for (r,1) and
                // (r,mu_horizon) - from which we can recover mu:

                // 視点 x から真下を向いたとき.
                const float d_min = atmosphere.top_radius - r;

                // 視点 x から地球に接する点までの距離は ρ で、その接点から大気の上端までの距離は H であるため、
                // 視点 x から大気の上端までの最大距離は ρ + H になる.
                const float d_max = rho + H;

                // uvwz.z に格納されているのは、u_μ. そして、大気の上端と交差する場合の、
                // 値の範囲は [0.5, 1.0] で、d_min に対応するのが 0.5 で、d_max に対応するのが 1.0 になるように、GetUnitRangeFromTextureCoord の引数を調整している.
                const float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                    2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);

                // d_min = 0 のときだけ d == 0 になり、それは、視点 x が大気の上端に接しているときで、そのときを mu = 1 に対応させる.
                // 余弦定理から、計算.
                // https://gemini.google.com/share/91af735989be
                mu = d == 0.0F
                    ? 1.0F
                    : aten::clamp((H * H - rho * rho - d * d) / (2.0F * r * d), -1.0F, 1.0F);
                ray_r_mu_intersects_ground = false;
            }

            const float x_mu_s =
                GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);

            // 大気層の距離の最小値.
            const float d_min = atmosphere.top_radius - atmosphere.bottom_radius;

            // 大気層の距離の最大値は、太陽が地平線にあるときの距離.
            const float d_max = H;

            const float D = DistanceToTopAtmosphereBoundary(
                atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
            const float A = (D - d_min) / (d_max - d_min);
            const float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
            const float d = d_min + min(a, A) * (d_max - d_min);

            // 余弦定理から、計算.
            // https://gemini.google.com/share/91af735989be
            mu_s = d == 0.0F
                ? 1.0F // ここまでの計算で、d = 0 になることはないはずだが、念のため、d = 0 のときは mu_s = 1 に対応させる.
                : aten::clamp((H * H - d * d) / (2.0F * atmosphere.bottom_radius * d), -1.0F, 1.0F);

            // [0,1] から [-1,1] への線形マッピング.
            nu = aten::clamp(uvwz.x * 2.0F - 1.0F, -1.0F, 1.0F);
        }
    }

    void GetRMuMuSNuFromScatteringTextureFragCoord(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::vec3& screen_coord,
        float& r,
        float& mu,
        float& mu_s,
        float& nu,
        bool& ray_r_mu_intersects_ground)
    {
        // https://gemini.google.com/share/3ec29779c23f
        // nu 以外は 0.5 オフセットしているが、ν はオフセットしていないので、SCATTERING_TEXTURE_NU_SIZE から 1 を引く.
        const vec4 SCATTERING_TEXTURE_SIZE = vec4(
            SCATTERING_TEXTURE_NU_SIZE - 1,
            SCATTERING_TEXTURE_MU_S_SIZE,
            SCATTERING_TEXTURE_MU_SIZE,
            SCATTERING_TEXTURE_R_SIZE);

        // ν、μs、μ、rの順でテクスチャに格納されていると仮定して、frag_coord.xからνとμsをアンパックします。
        const float screen_coord_nu = aten::floor(screen_coord.x / static_cast<float>(SCATTERING_TEXTURE_MU_S_SIZE));

        const float screen_coord_mu_s = aten::mod(screen_coord.x, static_cast<float>(SCATTERING_TEXTURE_MU_S_SIZE));

        vec4 uvwz(screen_coord_nu, screen_coord_mu_s, screen_coord.y, screen_coord.z);
        uvwz /= SCATTERING_TEXTURE_SIZE;

        _detail::GetRMuMuSNuFromScatteringTextureByTexCoord(
            atmosphere,
            screen_coord,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground);

        // Clamp nu to its valid range of values, given mu and mu_s.
        // 球面三角法と三角不等式から nu の値の範囲を求める.
        // https://gemini.google.com/share/da5dd355c041
        nu = aten::clamp(
            nu,
            mu * mu_s - sqrt((1.0F - mu * mu) * (1.0F - mu_s * mu_s)),
            mu * mu_s + sqrt((1.0F - mu * mu) * (1.0F - mu_s * mu_s)));
    }

    vec4 GetScatteringTextureUvwzFromRMuMuSNu(
        const aten::sky::AtmosphereParameters& atmosphere,
        const float r,
        const float mu,
        const float mu_s,
        const float nu,
        const bool ray_r_mu_intersects_ground)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);
        AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);
        AT_ASSERT(nu >= -1.0 && nu <= 1.0);

        // GetRMuMuSNuFromScatteringTextureUvwz の逆を行う.
        // GetRMuMuSNuFromScatteringTextureUvwz は、uvzw -> r, μ, μs, nu を取得していたが、
        // このAPIでは、r, μ, μs, nu -> uvzw を取得する.

        // Distance to top atmosphere boundary for a horizontal ray at ground level.
        // 論文中の 4. Precomputations の H = (Rt^2 - Rg^2)^1/2 に相当する値.
        // 地球のグラウンドレベルからの太陽が地平線にあるときの距離.
        const float H = aten::sqrt(atmosphere.top_radius * atmosphere.top_radius -
            atmosphere.bottom_radius * atmosphere.bottom_radius);

        // Distance to the horizon.
        const float rho = safe_sqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);

        const float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

        // Discriminant of the quadratic equation for the intersections of the ray
        // (r,mu) with the ground (see RayIntersectsGround).
        const float r_mu = r * mu;
        const float discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;

        float u_mu = 0.0F;

        if (ray_r_mu_intersects_ground) {
            // 論文の r・μ < 0 の場合. 視線レイが地面に交差する場合.

            // Distance to the ground for the ray (r,mu), and its minimum and maximum
            // values over all mu - obtained for (r,-1) and (r,mu_horizon).
            const float d = -r_mu - safe_sqrt(discriminant);
            const float d_min = r - atmosphere.bottom_radius;
            const float d_max = rho;
            u_mu = 0.5F - 0.5F * GetTextureCoordFromUnitRange(
                d_max == d_min
                ? 0.0F
                : (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
        }
        else {
            // 論文の otherwise の場合. 視線レイが地面に交差せず大気の境界に交差する場合.

            // Distance to the top atmosphere boundary for the ray (r,mu), and its
            // minimum and maximum values over all mu - obtained for (r,1) and
            // (r,mu_horizon).
            const float d = -r_mu + safe_sqrt(discriminant + H * H);
            const float d_min = atmosphere.top_radius - r;
            const float d_max = rho + H;
            u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
                (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
        }

        const float d = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, mu_s);
        const float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
        const float d_max = H;
        const float a = (d - d_min) / (d_max - d_min);
        const float D = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
        const float A = (D - d_min) / (d_max - d_min);
        // An ad-hoc function equal to 0 for mu_s = mu_s_min (because then d = D and
        // thus a = A), equal to 1 for mu_s = 1 (because then d = d_min and thus
        // a = 0), and with a large slope around mu_s = 0, to get more texture
        // samples near the horizon.
        const float u_mu_s = GetTextureCoordFromUnitRange(
            aten::max(1.0F - a / A, 0.0F) / (1.0F + a), SCATTERING_TEXTURE_MU_S_SIZE);

        const float u_nu = (nu + 1.0F) / 2.0F;

        return vec4(u_nu, u_mu_s, u_mu, u_r);
    }
}
