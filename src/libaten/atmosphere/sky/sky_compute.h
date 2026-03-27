#pragma once

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_coord_convert.h"
#include "atmosphere/sky/sky_params.h"

#include "math/vec3.h"

#include "image/texture.h"
#include "image/texture_3d.h"

// NOTE
// 計算単位
// - meter

namespace aten::sky {
    // 大気境界とのOpticalDepthを計算する.
    inline float ComputeOpticalLengthToTopAtmosphereBoundary(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::DensityProfile& profile,
        const float r,
        const float mu)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);

        // Number of intervals for the numerical integration.
        constexpr int32_t SAMPLE_COUNT = 500;

        // The integration step, i.e. the length of each integration interval.
        // 大気の上端境界までの距離をSAMPLE_COUNT等分する.
        const float dx = DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / static_cast<float>(SAMPLE_COUNT);

        // Integration loop.
        float result = 0.0F;

        for (int32_t i = 0; i <= SAMPLE_COUNT; ++i) {
            const float d_i = i * dx;

            // Distance between the current sample point and the planet center.
            // d_i での高度を計算.
            // |x + ts|^2 = r_i^2
            // => |x|^2 + 2tx・s + t^2|s|^2 = r_i^2
            // => r^2 + 2rμ + t^2 = r_i^2 (|x|=r, x・s=μ, |s|=1)
            // tについて解くと、t = -rμ ± sqrt(r^2(μ^2-1)+r_i^2) となるが、t = d_i となるため、
            // r_i^2 = d_i^2 + 2rμd_i + r^2 となる.
            const float r_i = aten::sqrt(d_i * d_i + 2.0F * r * mu * d_i + r * r);

            // atmosphere.bottom_radius : R_ground

            // Number density at the current sample point (divided by the number density
            // at the bottom of the atmosphere, yielding a dimensionless number).
            // Rayleigh散乱層、Mie散乱層の場合だと、exp(-h/H) を計算する.
            const float y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);

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

    namespace transmittance {
        inline aten::vec3 ComputeTransmittanceToTopAtmosphereBoundary(
            const aten::sky::AtmosphereParameters& atmosphere,
            const float r,
            const float mu)
        {
            AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
            AT_ASSERT(mu >= -1.0 && mu <= 1.0);

            // 式(5).
            // 論文より、βeR = βsR としているので、rayleigh_scattering を rayleigh_extinction として利用する.
            // absorption_extinction はオゾン層の吸収のための項.
            return aten::exp(-(
                atmosphere.rayleigh_scattering * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.rayleigh_density, r, mu)
                + atmosphere.mie_extinction * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.mie_density, r, mu)
                + atmosphere.absorption_extinction * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.absorption_density, r, mu)));
        }

        inline vec3 GetTransmittanceToTopAtmosphereBoundary(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& transmittance_texture,
            const float r,
            const float mu)
        {
            AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);

            // NOTE
            // #define DimensionlessSpectrum vec3

            // r, mu から、テクスチャのUV座標を計算して、テクスチャから値を読み取る.
            const auto uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
            return SampleTexture2D(transmittance_texture, uv);
        }

        // 視点xから視線ベクトル上のある点まで間の transmittance.
        // 視線ベクトル方向でのtransmittance.
        inline aten::vec3 GetTransmittance(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& transmittance_texture,
            const float r,
            const float mu,
            const float d,
            const bool ray_r_mu_intersects_ground)
        {
            AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
            AT_ASSERT(mu >= -1.0 && mu <= 1.0);
            AT_ASSERT(d >= 0.0);

            // 大気の上端境界（Rt）と下端境界（地球の半径:Rg）の間に収まるように、指定した距離をクランプする.
            // |x + dv|^2 = r_d^2 から、r_d を求める.
            // |x|^2 + 2dv・x + d^2|v|^2 = r_d^2
            //  => r^2 + 2rμd + d^2 = r_d^2
            //  => r_d = sqrt(d^2 + 2rμd + r^2)
            // となる.
            // そして、r_d が大気の上端境界と下端境界の間に収まるようにクランプする.
            const float r_d = aten::clamp(
                aten::sqrt(d * d + 2.0F * r * mu * d + r * r),
                atmosphere.bottom_radius,
                atmosphere.top_radius);

            // 地球の中心をO、r_d での点をQとすると、
            //  μ_d = OQ/|OQ|・v  = OQ ・v / r_d
            //   => μ_d = (OP + PQ)・v / r_d = (x・v + ds・v) / r_d
            //   => μ_d = (rμ + dν) / r_d
            const float mu_d = aten::clamp((r * mu + d) / r_d, -1.0F, 1.0F);

            // NOTE
            // #define DimensionlessSpectrum vec3

            // 式(5)から、transmittance は視点をx、大気上端の交点を i とすると、 exp(-∫_x^i).
            // また、途中の点を y とすると、 exp(-∫_x^i) = exp{-(∫_x^y + ∫_y^i}) = exp(-∫_x^y) * exp(-∫_y^i) となる.
            // つまり、exp(-∫_x^y) = exp(-∫_x^i) / exp(-∫_y^i) となるので、
            // transmittance(x,y) = transmittance(x,i) / transmittance(y,i) となる.
            // x ----- y ----- i

            if (ray_r_mu_intersects_ground) {
                // 視線レイが地面と交差する場合.
                // 視線レイが下方向なので、cos(π2 + θ) = -cos(θ) となるので、-mu_d、-mu をGetTransmittanceToTopAtmosphereBoundaryに渡す.

                // 地面に向かう場合、点 x (現在地) よりも 点 y (距離 d 先の点) の方が、「逆方向の大気上端」に近い状態になります.
                // - 視点のパス： Ground <- y <- x
                // - 計算に使うパス（逆向き）： x -> y -> TopAtmosphere
                // このとき、逆方向（大気上端方向）を基準に考えると：始点は x ではなく y になります。終点は y ではなく x になります.
                // つまり、逆向きのパスにおける「手前の点」が y で、「奥の点」が x になるため、
                // 割り算の順番も、GetTransmittanceToTopAtmosphereBoundary(y) / GetTransmittanceToTopAtmosphereBoundary(x) になります.
                // https://gemini.google.com/share/02e91c6f63f3

                const auto transmittance = GetTransmittanceToTopAtmosphereBoundary(
                    atmosphere, transmittance_texture, r_d, -mu_d) /
                    GetTransmittanceToTopAtmosphereBoundary(
                        atmosphere, transmittance_texture, r, -mu);
                return aten::vmin(transmittance, 1.0F);            }
            else {
                // 視線レイが大気と交差する場合.
                const auto transmittance = GetTransmittanceToTopAtmosphereBoundary(
                    atmosphere, transmittance_texture, r, mu) /
                    GetTransmittanceToTopAtmosphereBoundary(
                        atmosphere, transmittance_texture, r_d, mu_d);
                return aten::vmin(transmittance, 1.0F);
            }
        }

        // ある点から太陽方向に向かって大気上端までの間の transmittance.
        inline aten::vec3 GetTransmittanceToSun(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& transmittance_texture,
            const float r,
            const float mu_s)
        {
            // https://gemini.google.com/share/c76cb49c89eb

            // θh : 天頂（真上）から見た、地平線方向への角度（地平線天頂角）.
            const float sin_theta_h = atmosphere.bottom_radius / r;
            const float cos_theta_h = -aten::sqrt(aten::max(1.0F - sin_theta_h * sin_theta_h, 0.0F));

            // 太陽を点ではなく円盤として扱う.
            // 地平線からどのくらい太陽が見えているかによって、太陽方向へのtransmittanceを変化させたい.
            // つまり、太陽が全部見えている（地平線より完全に上）なら1.0、全く見えていない（地平線より完全に下）なら0.0、部分的に見えているなら 0 - 1 を補完した係数を掛けたい.
            // 見えている部分の面積（Circular Segment）を計算するとコストがかかるので、近似したい.
            // そこで、太陽方向への角度が太陽の見えている範囲の角度に収まっているかどうかで近似する.

            // αs : sun_angular_radius

            // 太陽上端 : cos(θh-αs)、太陽下端 : cos(θh+αs)
            // cos(θh±αs) = cosθh・cosαs ∓ sinθh・sinαs
            // αs は非常に小さい（αs<<1）ので、cosαs = 0、sinαs = αs に近似できる.
            // よって、cos(θh±αs) = cosθh ∓ αs・sinθh
            // cosθh - αs・sinθh <= μs <= cosθh + αs・sinθh
            //  => -αs・sinθh <= μs - cosθh <= αs・sinθh

            // GetTransmittanceToTopAtmosphereBoundary で mu_s 方向（太陽方向）の transmittance を取得.

            const auto transmittance = transmittance::GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, r, mu_s);

            const auto coeff = aten::smoothstep(
                    -sin_theta_h * atmosphere.sun_angular_radius,
                    sin_theta_h * atmosphere.sun_angular_radius,
                    mu_s - cos_theta_h);

            return transmittance * coeff;
        }
    }

    namespace irradiance {
        inline aten::vec3 GetIrradiance(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& irradiance_texture,
            const float r,
            const float mu_s)
        {
            vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
            return SampleTexture2D(irradiance_texture, uv);
        }
    }

    // 太陽からの入射放射照度を事前計算する.
    inline aten::vec3 ComputeDirectIrradiance(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const float r,
        const float mu_s)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);

        const float alpha_s = atmosphere.sun_angular_radius;

        /**
         [Explanation of the Average Cosine Factor Approximation]
         * 1. Small-Angle Approximation (Why mu_s can be compared with alpha_s):
         - mu_s = cos(zenith_angle). Near the horizon, mu_s = sin(elevation_angle).
         - Since the sun's angular radius (alpha_s) is very small (~0.0046 rad),
         we use the approximation: sin(x) $2248 x (in radians).
         - Therefore, mu_s effectively represents the sun's elevation in radians,
         allowing a direct comparison with alpha_s without expensive trig functions.

         2. Parabolic Smoothing (The formula: (mu + alpha)^2 / 4alpha):
         - This is a C1-continuous approximation of the visible fraction of the sun disc.
         - It smoothly interpolates between:
         a) mu_s < -alpha_s : 0.0 (Sun is completely below the horizon)
         b) mu_s >  alpha_s : mu_s (Sun is completely above the horizon)
         - The quadratic form f(mu) = (mu + alpha)^2 / (4 * alpha) ensures:
         - f(-alpha) = 0
         - f(alpha)  = alpha (matches mu_s at the boundary)
         - Derivatives (f') match at both boundaries (0 and 1), preventing visual
         artifacts like Mach bands during sunset/sunrise.

         3. Efficiency:
         - Avoids expensive geometric area calculations (involving acos/sqrt)
         while maintaining physical plausibility for atmospheric scattering.
        */
        const float average_cosine_factor = mu_s < -alpha_s
            ? 0.0F   // 太陽全体が地平線の下にある場合
            : (mu_s > alpha_s
                ? mu_s  // 太陽全体が地平線の上にある場合
                : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0F * alpha_s));   // 太陽が地平線の近くにある場合に 0 から α_s までがなめらかに変化するように近似.

        // https://gemini.google.com/share/e287cbd05533

        // Algorithm 4.1 の最初の ΔE を計算するための式.
        // 式(15)を計算. 最初なので、L* を L0 として、
        // L0 = Lsun * T(x0, 大気の上端)

        // なんで、積分でないのか？
        //   太陽光（Direct Irradiance）に限って言えば、光は太陽のある一点（方向 $\mathbf{s}$）からしか来ない.
        //   数学的には、太陽の輝度分布は方向 s 以外では 0 になる「デルタ関数」として扱うことができる.
        //   積分の性質: 関数にデルタ関数を掛けて積分すると、その一点での値を取り出すことになる.
        //   結果: 積分記号が消え、単純な「太陽の輝度 × 透過率 × 投影角（cosθ）」という掛け算のみになる.

        // 論文の式(15)は、主に地表における反射を説明するために記述されてる.
        // しかし、Brunetonのアルゴリズムでは以下の理由から、地表以外の高度 r における放射照度も必要になる.
        //    - 多重散乱（Multiple Scattering）の計算
        //        ある高度 rr にある空気分子が、太陽から直接届く光（Direct Irradiance）を散乱して、さらに別の方向へ飛ばすプロセスを計算する必要がある.
        //        このとき、その高度 r での光の強さを知る必要がある.
        //    - 空中にある物体のライティング:
        //        地表にいない物体（例えば飛行機や雲）をレンダリングする場合、その高度 r における太陽光の減衰を考慮した放射照度が必要になる.

        // これが地面からの反射として、利用されるかどうかは後段で別途判定され、処理される.

        const auto transmittance = transmittance::GetTransmittanceToTopAtmosphereBoundary(
            atmosphere,
            transmittance_texture,
            r, mu_s);

        return atmosphere.solar_irradiance * transmittance * average_cosine_factor;
    }

    namespace single_scattering {
        inline void ComputeSingleScatteringIntegrand(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& transmittance_texture,
            const float r,
            const float mu,
            const float mu_s,
            const float nu,
            const float d,
            const bool ray_r_mu_intersects_ground,
            aten::vec3& rayleigh,
            aten::vec3& mie)
        {

            // 大気の上端境界（Rt）と下端境界（地球の半径:Rg）の間に収まるように、指定した距離をクランプする.
            // |x + dv|^2 = r_d^2 から、r_d を求める.
            // |x|^2 + 2dv・x + d^2|v|^2 = r_d^2
            //  => r^2 + 2rμd + d^2 = r_d^2
            //  => r_d = sqrt(d^2 + 2rμd + r^2)
            // となる.
            // そして、r_d が大気の上端境界と下端境界の間に収まるようにクランプする.
            const float r_d = aten::clamp(
                aten::sqrt(d * d + 2.0F * r * mu * d + r * r),
                atmosphere.bottom_radius,
                atmosphere.top_radius);

            // 地球の中心をO、r_d での点をQとしたときの、点Qでの太陽方向への μ_s_d を計算.
            //  μ_s_d = OQ/|OQ|・s = OQ ・s / r_d
            //   => μ_s_d = (OP + PQ)・s / r_d = (x・s + ds・v) / r_d
            //   => μ_s_d = (rμ_s + dν) / r_d
            const float mu_s_d = aten::clamp((r * mu_s + d * nu) / r_d, -1.0F, 1.0F);

            // 太陽 -> x+dv で表される点Q -> 視点P という経路のtotalのtransmittanceを計算したい.
            //  T_total = T_{q->sun} × T_{p->q}
            // これにより、太陽 -> Q -> P の経路の transmittance になる.
            aten::vec3 transmittance{
                transmittance::GetTransmittance(
                    atmosphere, transmittance_texture, r, mu, d,
                    ray_r_mu_intersects_ground)
            };
            transmittance *= transmittance::GetTransmittanceToSun(
                atmosphere, transmittance_texture, r_d, mu_s_d);

            // Rayleigh散乱層、Mie散乱層の場合だと、density: exp(-h/H) を計算する.
            // http://nishitalab.org/user/nis/cdrom/sig93_nis.pdf
            // 式(8) で density * transmittance を積分している.
            rayleigh = transmittance * GetProfileDensity(
                atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
            mie = transmittance * GetProfileDensity(
                atmosphere.mie_density, r_d - atmosphere.bottom_radius);
        }

        inline void ComputeSingleScattering(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture& transmittance_texture,
            const float r,
            const float mu,
            const float mu_s,
            const float nu,
            const bool ray_r_mu_intersects_ground,
            aten::vec3& rayleigh,
            aten::vec3& mie)
        {
            AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
            AT_ASSERT(mu >= -1.0 && mu <= 1.0);
            AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);
            AT_ASSERT(nu >= -1.0 && nu <= 1.0);

            // Number of intervals for the numerical integration.
            const int SAMPLE_COUNT = 50;

            // The integration step, i.e. the length of each integration interval.
            // 地球 or 大気の境界までの距離を計算し、それをSAMPLE_COUNT等分する.
            // ray_r_mu_intersects_ground によって、地球 or 大気の境界のどちらになるか決まる.
            const float dx =
                DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,
                    ray_r_mu_intersects_ground) / static_cast<float>(SAMPLE_COUNT);

            // Integration loop.
            aten::vec3 rayleigh_sum{ 0.0F };
            aten::vec3 mie_sum{ 0.0F };

            // NOTE:
            // 太陽光の１回目の single scattering なので、光は全方向から来ず、s 方向からしか光はこないので、全球積分はここでは必要ない.

            for (int32_t i = 0; i <= SAMPLE_COUNT; ++i) {
                const float d_i = i * dx;

                // The Rayleigh and Mie single scattering at the current sample point.
                aten::vec3 rayleigh_i;
                aten::vec3 mie_i;
                ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture,
                    r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);

                // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
                // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
                //   = (y0/2 + y1 + y2 + y3/2) * dx
                // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

                // Sample weight (from the trapezoidal rule).
                float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5F : 1.0F;

                // 台形公式による積分として、本来はここで dx を掛けるべきだが
                // ループの外で1回だけ dx を掛けることで計算コストを減らしている.
                rayleigh_sum += rayleigh_i * weight_i;
                mie_sum += mie_i * weight_i;
            }

            rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *
                atmosphere.rayleigh_scattering;
            mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
        }
    }

    namespace scattering {
        inline aten::vec3 GetScattering(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture3d& scattering_texture,
            const float r,
            const float mu,
            const float mu_s,
            const float nu,
            const bool ray_r_mu_intersects_ground)
        {
            // r, μ, μs, nu -> uvzw を取得する.
            vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
                atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            // https://gemini.google.com/share/aad15c090df3

            // ３次元テクスチャでは４成分をどうしても格納しきれないので、
            // x成分に μs と nu を格納している.
            // （詳細は GetRMuMuSNuFromScatteringTextureFragCoord ）
            // どうしｔも nu の精度は低くなるので、lerp することで、精度を上げている.

            const float tex_coord_x = uvwz.x * static_cast<float>(SCATTERING_TEXTURE_NU_SIZE - 1);
            const float tex_x = floor(tex_coord_x);
            const float lerp = tex_coord_x - tex_x;
            vec3 uvw0{ (tex_x + uvwz.y) / static_cast<float>(SCATTERING_TEXTURE_NU_SIZE),
                uvwz.z, uvwz.w };
            vec3 uvw1{ (tex_x + 1.0 + uvwz.y) / static_cast<float>(SCATTERING_TEXTURE_NU_SIZE),
                uvwz.z, uvwz.w };

            const auto a{ SampleTexture3D(scattering_texture, uvw0) };
            const auto b{ SampleTexture3D(scattering_texture, uvw1) };
            return a * (1.0F - lerp) + b * lerp;
        }

        inline aten::vec3 GetScattering(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::texture3d& single_rayleigh_scattering_texture,
            const aten::texture3d& single_mie_scattering_texture,
            const aten::texture3d& multiple_scattering_texture,
            const float r,
            const float mu,
            const float mu_s,
            const float nu,
            const bool ray_r_mu_intersects_ground,
            const int32_t scattering_order)
        {
            if (scattering_order == 1) {
                aten::vec3 rayleigh{ GetScattering(
                    atmosphere, single_rayleigh_scattering_texture, r, mu, mu_s, nu,
                    ray_r_mu_intersects_ground) };
                aten::vec3 mie{ GetScattering(
                    atmosphere, single_mie_scattering_texture, r, mu, mu_s, nu,
                    ray_r_mu_intersects_ground) };
                return rayleigh * RayleighPhaseFunction(nu) +
                    mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
            }
            else {
                return GetScattering(
                    atmosphere, multiple_scattering_texture, r, mu, mu_s, nu,
                    ray_r_mu_intersects_ground);
            }
        }
    }

    inline aten::vec3 ComputeScatteringDensity(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        const aten::texture& irradiance_texture,
        const float r,
        const float mu,
        const float mu_s,
        const float nu,
        const int32_t scattering_order)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu >= -1.0 && mu <= 1.0);
        AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);
        AT_ASSERT(nu >= -1.0 && nu <= 1.0);
        AT_ASSERT(scattering_order >= 2);

        // Compute unit direction vectors for the zenith, the view direction omega and
        // and the sun direction omega_s, such that the cosine of the view-zenith
        // angle is mu, the cosine of the sun-zenith angle is mu_s, and the cosine of
        // the view-sun angle is nu. The goal is to simplify computations below.

        // 計算の基準となる上向きのベクトル（天頂）を Z軸に固定.
        const aten::vec3 zenith_direction{ 0.0F, 0.0F, 1.0F };

        // 視線ベクトル.
        // ω を XZ平面上に置くように設定（Y=0）. これにより後の計算を簡略化.
        // z成分 が μ、x成分は ω が正規化済みベクトルとして y, z成分から計算.
        const  aten::vec3 omega{ aten::sqrt(1.0F - mu * mu), 0.0F, mu };

        // 太陽方向ベクトル.
        // nu = v・s = ω・ωs = (ω_x, ω_y, ω_z)・(ωs_x, ωs_y, ωs_z) を展開.
        // nu = ω_x・ωs_x + ω_y・ωs_y + ω_z・ωs_z
        //    = ω_x・ωs_x + 0・ωs_y + μ・μs  (上記より、ω_y = 0, ω_z = μ となり、視線ベクトルと同様に太陽方向ベクトルでも ω_z = μs)
        // ここで、ωs_x について解くと
        //    ωs_x = (nu - μ・μs) / ω_x
        // z成分が μs、x成分が計算済みで、ωs が正規化済みベクトルとして、y成分は x、z成分から計算.
        const float sun_dir_x = omega.x == 0.0F ? 0.0F : (nu - mu * mu_s) / omega.x;
        const float sun_dir_y = aten::sqrt(aten::max(1.0F - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0F));
        aten::vec3 omega_s{ sun_dir_x, sun_dir_y, mu_s };

        constexpr int32_t SAMPLE_COUNT = 16;
        const float dphi = AT_MATH_PI / static_cast<float>(SAMPLE_COUNT);
        const float dtheta = AT_MATH_PI / static_cast<float>(SAMPLE_COUNT);

        // 0で初期化.
        aten::vec3 rayleigh_mie{ 0.0F };

        // Nested loops for the integral over all the incident directions omega_i.
        // 全球積分をしていく.
        for (int32_t l = 0; l < SAMPLE_COUNT; ++l) {
            const float theta = (static_cast<float>(l) + 0.5F) * dtheta;
            const float cos_theta = aten::cos(theta);
            const float sin_theta = aten::sin(theta);

            // 入ってくるレイの先が地面と交差するかをチェック.
            // もし、地面と交差するなら、地面からの反射として扱う.
            const bool is_ray_r_theta_intersects_ground =
                RayIntersectsGround(atmosphere, r, cos_theta);

            // The distance and transmittance to the ground only depend on theta, so we
            // can compute them in the outer loop for efficiency.
            float distance_to_ground = 0.0;
            aten::vec3 transmittance_to_ground{ 0.0F };
            aten::vec3 ground_albedo{ 0.0F };

            if (is_ray_r_theta_intersects_ground) {
                // 地面からの反射を計算.
                distance_to_ground = DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);
                transmittance_to_ground = transmittance::GetTransmittance(
                    atmosphere, transmittance_texture,
                    r, cos_theta,
                    distance_to_ground,
                    true /* ray_intersects_ground */);
                ground_albedo = atmosphere.ground_albedo;
            }

            // φ を [0,2pi] で積分する.
            // dphi は pi/SAMPLE_COUNT なので、[0, 2pi] になるためには、2 * SAMPLE_COUNT 回ループしないといけない.
            for (int32_t m = 0; m < 2 * SAMPLE_COUNT; ++m) {
                const float phi = (static_cast<float>(m) + 0.5F) * dphi;

                // 入ってくるベクトルを計算.
                vec3 omega_i{ cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta };

                // NOTE
                // const SolidAngle sr = 1.0;

                // 積分の極小立体角（sinθdθdφ）を計算.
                const float domega_i = dtheta * dphi * sin(theta);

                // 入ってくる raddianceを取得.
                // 初回（scattering_order - 1 == 1）の場合は、入射した太陽光をレイリー散乱、ミー散乱させたもの.
                // 二回目以降は、multiple scattering して入射してきたもの.

                // The radiance L_i arriving from direction omega_i after n-1 bounces is
                // the sum of a term given by the precomputed scattering texture for the
                // (n-1)-th order:
                const float nu1 = aten::dot(omega_s, omega_i);
                aten::vec3 incident_radiance{
                    scattering::GetScattering(
                        atmosphere,
                        single_rayleigh_scattering_texture,
                        single_mie_scattering_texture,
                        multiple_scattering_texture,
                        r,
                        omega_i.z, // μ
                        mu_s,
                        nu1,
                        is_ray_r_theta_intersects_ground,
                        scattering_order - 1)
                };

                // 以下で地面からの反射を計算.
                // ただし、レイが地面と交差していない場合は、
                // transmittance_to_ground、ground_albedo はゼロのままで、反射の計算結果もゼロになる.

                // and of the contribution from the light paths with n-1 bounces and whose
                // last bounce is on the ground. This contribution is the product of the
                // transmittance to the ground, the ground albedo, the ground BRDF, and
                // the irradiance received on the ground after n-2 bounces.
                vec3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);

                const aten::vec3 ground_irradiance{
                    irradiance::GetIrradiance(
                        atmosphere,
                        irradiance_texture,
                        atmosphere.bottom_radius,
                        aten::clamp(dot(ground_normal, omega_s), -1.0F, 1.0F))
                };
                incident_radiance += transmittance_to_ground *
                    ground_albedo * (1.0F / AT_MATH_PI) * ground_irradiance;

                // The radiance finally scattered from direction omega_i towards direction
                // -omega is the product of the incident radiance, the scattering
                // coefficient, and the phase function for directions omega and omega_i
                // (all this summed over all particle types, i.e. Rayleigh and Mie).
                const float nu2 = dot(omega, omega_i);
                const float rayleigh_density = GetProfileDensity(
                    atmosphere.rayleigh_density, r - atmosphere.bottom_radius);
                const float mie_density = GetProfileDensity(
                    atmosphere.mie_density, r - atmosphere.bottom_radius);

                // 式(7)の被積分部分.
                rayleigh_mie += incident_radiance * (
                    atmosphere.rayleigh_scattering * rayleigh_density *
                    RayleighPhaseFunction(nu2) +
                    atmosphere.mie_scattering * mie_density *
                    MiePhaseFunction(atmosphere.mie_phase_function_g, nu2)) *
                    domega_i;
            }
        }
        return rayleigh_mie;
    }

    // 太陽光 **でない** 光からの放射照度を計算.
    inline aten::vec3 ComputeIndirectIrradiance(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        const float r,
        const float mu_s,
        const int32_t scattering_order)
    {
        AT_ASSERT(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
        AT_ASSERT(mu_s >= -1.0 && mu_s <= 1.0);
        AT_ASSERT(scattering_order >= 1);

        // 論文内での式(12)を計算.
        // 地面でない点での計算になるが、計算結果が使われるのは時間からの反射の計算時なので、問題ない.
        // ComputeDirectIrradiance でも、地面でない点についても計算していたので、そのあたりは基本的に同じ考え方になる.

        constexpr int32_t SAMPLE_COUNT = 32;
        const float dphi = AT_MATH_PI / static_cast<float>(SAMPLE_COUNT);
        const float dtheta = AT_MATH_PI / static_cast<float>(SAMPLE_COUNT);

        // ゼロで初期化.
        aten::vec3 result{ 0.0F };

        // ComputeScatteringDensity でやったように z 方向を基準にして、太陽方向へのベクトルを計算.
        // y = 0 にすることで、後の計算を簡易化.
        aten::vec3 omega_s{ aten::sqrt(1.0F - mu_s * mu_s), 0.0F, mu_s };

        // 半球積分.
        for (int32_t j = 0; j < SAMPLE_COUNT / 2; ++j) {
            const float theta = (j + 0.5F) * dtheta;

            for (int32_t i = 0; i < 2 * SAMPLE_COUNT; ++i) {
                const float phi = (i + 0.5F) * dphi;

                // 入射ベクトル.
                const vec3 omega{ cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta) };

                // 積分の極小立体角: dω = sinθdθdφ
                const float domega = dtheta * dphi * sin(theta);

                const float nu = dot(omega, omega_s);

                // omega.z = cosθ なので、omega.z はそのまま consine factor になる.

                // 最終的には、地面からの反射を計算するときに使うし、点の上側の半球積分なので
                // そもそも地面と交差する場合を考える必要はない.
                const auto scattering{
                    scattering::GetScattering(
                        atmosphere,
                        single_rayleigh_scattering_texture,
                        single_mie_scattering_texture,
                        multiple_scattering_texture,
                        r,
                        omega.z,
                        mu_s,
                        nu,
                        false /* ray_r_theta_intersects_ground */,
                        scattering_order)
                };
                result += scattering * omega.z * domega;
            }
        }
        return result;
    }

    inline aten::vec3 ComputeMultipleScattering(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& scattering_density_texture,
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

        // Number of intervals for the numerical integration.
        constexpr int32_t SAMPLE_COUNT = 50;

        // The integration step, i.e. the length of each integration interval.
        // 大気の境界 or 地球に交差する点までの距離を計算し、step 数で除算.
        const float dx =
            DistanceToNearestAtmosphereBoundary(
                atmosphere, r, mu, ray_r_mu_intersects_ground) /
            static_cast<float>(SAMPLE_COUNT);

        // Integration loop.
        // ゼロで初期化.
        aten::vec3 rayleigh_mie_sum{ 0.0F };

        for (int32_t i = 0; i <= SAMPLE_COUNT; ++i) {
            const float d_i = static_cast<float>(i) * dx;

            // The r, mu and mu_s parameters at the current integration point (see the
            // single scattering section for a detailed explanation).

            // |x + d_i・v|^2 = r_i^2
            //  => |x|^2 + 2d_i・x・v + d_i^2|v|^2 = r_i^2
            //  => r^2 + 2d_i・rμ + d_i^2 = r_i^2
            // r_i について、解くと
            //  r_i = sqrt(d_i^2 + 2d_i・rμ + r^2)
            const float r_i = aten::clamp(
                aten::sqrt(d_i * d_i + 2.0F * r * mu * d_i + r * r),
                atmosphere.bottom_radius,
                atmosphere.top_radius);

            // 地球の中心をO、開始点をP、r_i での点をQとすると、
            //  μ_i = OQ/|OQ|・v = OQ・v / r_i
            //      = (OP + PQ)・v / r_i = (OP・v + PQ・v) / r_i
            //      = (x・v + d_i・v・v) / r_i
            //      = (rμ + d_i|v|^2) / r_i = (rμ + d_i) / r_i
            const float mu_i = aten::clamp((r * mu + d_i) / r_i, -1.0F, 1.0F);

            // 地球の中心をO、開始点をP、r_i での点をQとすると、
            // μs_i = OQ/|OQ|・s = OQ・s / r_i
            //      = (OP + PQ)・s / r_i = (OP・s + PQ・s) / r_i
            //      = (x・s + d_i・v・s) / r_i
            //      = (rμs + d_i・nu) / r_i
            const float mu_s_i = aten::clamp((r * mu_s + d_i * nu) / r_i, -1.0F, 1.0F);

            // The Rayleigh and Mie multiple scattering at the current sample point.
            aten::vec3 rayleigh_mie_i{
                scattering::GetScattering(
                    atmosphere, scattering_density_texture, r_i, mu_i, mu_s_i, nu,
                    ray_r_mu_intersects_ground) *
                transmittance::GetTransmittance(
                    atmosphere, transmittance_texture, r, mu, d_i,
                    ray_r_mu_intersects_ground) *
                dx
            };

            // 台形公式による積分の場合、例えば、分割数3で単純に計算すると、
            // (y0 + y1) * dx / 2 + (y1 + y2) * dx / 2 + (y2 + y3) * dx / 2
            //   = (y0/2 + y1 + y2 + y3/2) * dx
            // となる. つまり、i=0とi=SAMPLE_COUNTのときは、y_iの重みが0.5で、それ以外のときは1.0と計算することもできる.

            // Sample weight (from the trapezoidal rule).
            const float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5F : 1.0F;
            rayleigh_mie_sum += rayleigh_mie_i * weight_i;
        }

        return rayleigh_mie_sum;
    }
}
