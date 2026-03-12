#pragma once

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"
#include "atmosphere/sky/sky_render.h"

#include "camera/pinhole.h"

#include "math/vec3.h"

#include "misc/tuple.h"

#include "image/texture.h"
#include "image/texture_3d.h"

//#pragma optimize("", off)

namespace aten::sky {
    // irradiance to radiance
    // radiance は 単位面積当たりではなく、立体角方向に面積を投影した、投影単位面積あたりの値.
    // 太陽の視半径は r/a (r: 天体の半径, a: 天体までの距離) から導出される
    // 天体までの距離 1 に対する比率となる.
    // そこから立体角を計算したときに ω = S / r^2
    // r = 1 となる.
    // 今回の場合は、太陽を円盤として立体角は
    //  S = pi x radius ^ 2
    inline AT_DEVICE_API aten::vec3 GetSolarRadiance(const aten::sky::AtmosphereParameters& atmosphere)
    {
        const auto sun_solid_angle = AT_MATH_PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius;
        return atmosphere.solar_irradiance / sun_solid_angle;
    }

    namespace {
        inline AT_DEVICE_API aten::tuple<aten::vec3, aten::vec3> GetCombinedScattering(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::PreComputeTextures& texture,
            const float r,
            const float mu,
            const float mu_s,
            const float nu,
            const bool ray_r_mu_intersects_ground)
        {
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
            const aten::vec3 uvw0{
                (tex_x + uvwz.y) / static_cast<float>(SCATTERING_TEXTURE_NU_SIZE),
                uvwz.z,
                uvwz.w
            };
            const aten::vec3 uvw1{
                (tex_x + 1.0 + uvwz.y) / static_cast<float>(SCATTERING_TEXTURE_NU_SIZE),
                uvwz.z,
                uvwz.w
            };

#ifdef COMBINED_SCATTERING_TEXTURES
            // TODO
            static_assert(false);
#else
            const auto scattering_0{ aten::sky::SampleTexture3D(texture.scattering_texture, uvw0) };
            const auto scattering_1{ aten::sky::SampleTexture3D(texture.scattering_texture, uvw1) };
            const auto scattering{ scattering_0 * (1.0F - lerp) + scattering_1 * lerp };

            const auto single_mie_scattering_0{ aten::sky::SampleTexture3D(texture.optional_single_mie_scattering_texture, uvw0) };
            const auto single_mie_scattering_1{ aten::sky::SampleTexture3D(texture.optional_single_mie_scattering_texture, uvw1) };
            const auto single_mie_scattering{
                single_mie_scattering_0 * (1.0F - lerp) + single_mie_scattering_1 * lerp
            };
#endif

            return aten::make_tuple(scattering, single_mie_scattering);
        }

    }

    /*
        To render the sky we simply need to display the sky radiance, which we can
        get with a lookup in the precomputed scattering texture(s), multiplied by the
        phase function terms that were omitted during precomputation.We can also return
        the transmittance of the atmosphere(which we can get with a single lookup in
        the precomputed transmittance texture), which is needed to correctly render the
        objects in space(such as the Sun and the Moon).This leads to the following
        function, where most of the computations are used to correctly handle the case
        of viewers outside the atmosphere, and the case of light shafts:
    */
    inline AT_DEVICE_API aten::vec3 GetSkyRadiance(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& camera,
        const aten::vec3& view_ray,
        const float shadow_length,
        const aten::vec3& sun_direction,
        aten::vec3& out_transmittance)   // 視線の先に太陽が見えているときに太陽光の減衰を計算する用.
    {
        // Compute the distance to the top atmosphere boundary along the view ray,
        // assuming the viewer is in space (or NaN if the view ray does not intersect
        // the atmosphere).
        float r = length(camera);
        float rmu = dot(camera, view_ray);

        // 大気の境界までの距離.
        // |x + tv|^2 = Rt^2
        //  => |x|^2 + 2tx・v + t^2|v|^2 = Rt^2
        //  => r^2 + 2trμ + t^2 = Rt^2
        // tの２次方程式として：
        //  t^2 + 2rμt + r^2 - Rt^2 = 0
        // これを解くと:
        //  t = -rμ ± sqrt((rμ)^2 - (r^2 - Rt^2))
        //    = -rμ ± sqrt((rμ)^2 - r^2 + Rt^2)
        // 視線から近いほうの交差点が見えている点なので、小さいほうを採用して、
        //  t = -rμ - sqrt((rμ)^2 - r^2 + Rt^2)
        const float distance_to_top_atmosphere_boundary = -rmu -
            aten::sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);

        auto camera_pos{ camera };

        // If the viewer is in space and the view ray intersects the atmosphere, move
        // the viewer to the top atmosphere boundary (along the view ray):
        // 視線が大気の境界と交差するが視点が大気の外にある場合は、視線に沿った大気の境界の上端に視点を移動させる.
        if (distance_to_top_atmosphere_boundary > 0.0F) {
            camera_pos = camera + view_ray * distance_to_top_atmosphere_boundary;
            r = atmosphere.top_radius;
            rmu += distance_to_top_atmosphere_boundary;
        }
        else if (r > atmosphere.top_radius) {
            // If the view ray does not intersect the atmosphere, simply return 0.

            // 視線が大気の境界と交差もせず、視点が大気の外にある場合.

            // 大気の外から太陽を見ても、大気による減衰が発生しないので、tranmittance は減衰しないように 1 にしておく.
            out_transmittance = aten::vec3(1.0F);

            // 視線が大気の境界と交差しないので、そもそも大気が見えないから、大気を描画することがないので、大気の radiance もゼロでいい.
            return aten::vec3(0.0F);
        }

        // Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
        const float mu = rmu / r;
        const float mu_s = dot(camera_pos, sun_direction) / r;
        const float nu = dot(view_ray, sun_direction);
        const bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

        // 大気の境界までの transmittace を使うことになる.
        // v ---------- atmosphere bound ---------> sun
        // |<-----transmittance-->| no transmittance  |
        out_transmittance = ray_r_mu_intersects_ground
            ? aten::vec3(0.0F)
            : aten::sky::transmittance::GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, texture.transmittance_texture, r, mu);

        aten::vec3 single_mie_scattering{ 0.0F };
        aten::vec3 scattering{ 0.0F };

        if (shadow_length == 0.0F) {
            aten::tie(scattering, single_mie_scattering) = GetCombinedScattering(
                atmosphere, texture,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);
        }
        else {
            // TODO
            AT_ASSERT(false);
        }

        return scattering * RayleighPhaseFunction(nu)
            + single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
    }

    inline AT_DEVICE_API aten::vec3 RenderSky(
        int32_t x, int32_t y,
        const aten::CameraParameter& camera,
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& sun_radiance_to_luminance,
        const aten::vec3& sky_radiance_to_luminance,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center,
        const float sun_size)
    {
        const float s = x / static_cast<float>(camera.width);
        const float t = y / static_cast<float>(camera.height);

        // TODO
        // Pinhole?
        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        const auto camera_org{ camsample.r.org };
        const auto view_direction{ camsample.r.dir };

        // TODO
        const float shadow_length = 0.0F;

        aten::vec3 transmittance{ 0.0F };
        aten::vec3 radiance{
            GetSkyRadiance(
                atmosphere, texture,
                camera_org - earth_center,
                view_direction,
                shadow_length,
                sun_direction,
                transmittance)
        };

        radiance = sky_radiance_to_luminance * radiance;

        // If the view ray intersects the Sun, add the Sun radiance.
        // ここで、視線方向のベクトルを v、太陽の方向ベクトルを s とします（どちらも単位ベクトル）.
        // この2つのベクトルのなす角を α とすると、視線が太陽の円盤内にある条件は α <= θ です.
        // これを、内積 cosα を使って判定する場合、条件は cosα >= cosθ となります。
        // sun_size.y = cos(SunAngularRadius) なので、dot することで cosine で比較する.
        // cosθ は θ が 0 に近いほど大きくなる.
        // つまり、view_direction と sun_direction が近いほど、値は大きくなる.
        // なので、>（大なり）だと太陽の視半径内といえる.
        if (dot(view_direction, sun_direction) > sun_size) {
            auto solar_radiance{ GetSolarRadiance(atmosphere) };
            solar_radiance = sun_radiance_to_luminance * solar_radiance;

            radiance = radiance + transmittance * solar_radiance;
        }

        return radiance;
    }

    namespace {
        inline aten::tuple<float, float, float> CieColorMatchingFunctionTableValue(const int32_t wavelength)
        {
            if (wavelength <= aten::sky::LambdaMin || wavelength >= aten::sky::LambdaMax) {
                return aten::make_tuple(0.0F, 0.0F, 0.0F);
            }

            // wavelength in function table is defined per 5.0 nm.
            auto u = (wavelength - aten::sky::LambdaMin) / 5.0F;

            const auto row = static_cast<int32_t>(aten::floor(u));
            AT_ASSERT(row >= 0 && row + 1 < CIE_2_DEG_COLOR_MATCHING_FUNCTIONS.size());
            AT_ASSERT(CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row].labmda <= wavelength
                && CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row + 1].labmda >= wavelength);

            const auto& element_0 = CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row];
            const auto& element_1 = CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[row + 1];

            u -= row;

            const auto x = aten::lerp(element_0.x, element_1.x, u);
            const auto y = aten::lerp(element_0.y, element_1.y, u);
            const auto z = aten::lerp(element_0.z, element_1.z, u);

            return aten::make_tuple(x, y, z);
        }
    }

    /*
        We can then implement a utility function to compute the "spectral radiance to luminance" conversion constants
        (see Section 14.3 in https://arxiv.org/pdf/1612.04336.pdf"A Qualitative and Quantitative Evaluation of 8 Clear Sky Models</a> for their definitions):
    */
    // The returned constants are in lumen.nm / watt.
    inline aten::vec3 ComputeSpectralRadianceToLuminanceFactors(
        const std::vector<float>& wavelengths,
        const std::vector<float>& solar_irradiance,
        const float lambda_power)
    {
        float k_r = 0.0F;
        float k_g = 0.0F;
        float k_b = 0.0F;

        const float solar_r = InterpolateFactor(wavelengths, solar_irradiance, aten::sky::LambdaR);
        const float solar_g = InterpolateFactor(wavelengths, solar_irradiance, aten::sky::LambdaG);
        const float solar_b = InterpolateFactor(wavelengths, solar_irradiance, aten::sky::LambdaB);

        constexpr int32_t dlambda = 1;

        for (int32_t lambda = aten::sky::LambdaMin; lambda < aten::sky::LambdaMax; lambda += dlambda)
        {
            float x_bar, y_bar, z_bar;
            aten::tie(x_bar, y_bar, z_bar) = CieColorMatchingFunctionTableValue(lambda);

            const auto r_bar = XYZ_TO_SRGB[0] * x_bar + XYZ_TO_SRGB[1] * y_bar + XYZ_TO_SRGB[2] * z_bar;
            const auto g_bar = XYZ_TO_SRGB[3] * x_bar + XYZ_TO_SRGB[4] * y_bar + XYZ_TO_SRGB[5] * z_bar;
            const auto b_bar = XYZ_TO_SRGB[6] * x_bar + XYZ_TO_SRGB[7] * y_bar + XYZ_TO_SRGB[8] * z_bar;

            const auto irradiance = InterpolateFactor(wavelengths, solar_irradiance, static_cast<float>(lambda));

            k_r += r_bar * irradiance / solar_r * aten::pow(lambda / aten::sky::LambdaR, lambda_power);
            k_g += g_bar * irradiance / solar_g * aten::pow(lambda / aten::sky::LambdaG, lambda_power);
            k_b += b_bar * irradiance / solar_b * aten::pow(lambda / aten::sky::LambdaB, lambda_power);
        }

        k_r *= aten::sky::MAX_LUMINOUS_EFFICACY * dlambda;
        k_g *= aten::sky::MAX_LUMINOUS_EFFICACY * dlambda;
        k_b *= aten::sky::MAX_LUMINOUS_EFFICACY * dlambda;

        return { k_r, k_g, k_b };
    }

    inline aten::vec3 ConvertSpectrumToLinearSrgb(
        const std::vector<float>& wavelengths,
        const std::vector<float>& spectrum)
    {
        constexpr int32_t dlambda = 1;

        // Convert the spectrum to XYZ color space.
        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;

        for (int32_t lambda = LambdaMin; lambda < LambdaMax; lambda += dlambda)
        {
            const float value = InterpolateFactor(spectrum, wavelengths, lambda);

            float _x, _y, _z;
            aten::tie(_x, _y, _z) = CieColorMatchingFunctionTableValue(lambda);
            x += _x * value;
            y += _y * value;
            z += _z * value;
        }
        const auto r = MAX_LUMINOUS_EFFICACY * (XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z) * dlambda;
        const auto g = MAX_LUMINOUS_EFFICACY * (XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z) * dlambda;
        const auto b = MAX_LUMINOUS_EFFICACY * (XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z) * dlambda;

        return aten::vec3(r, g, b);
    }

}
