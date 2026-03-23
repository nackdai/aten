#pragma once

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"

#include "math/vec3.h"

#include "misc/tuple.h"

#include "image/texture.h"
#include "image/texture_3d.h"

namespace aten::sky {
    // irradiance to radiance
    // radiance は 単位面積当たりではなく、立体角方向に面積を投影した、投影単位面積あたりの値.
    // 太陽の視半径は r/a (r: 天体の半径, a: 天体までの距離) から導出される
    // 天体までの距離 1 に対する比率となる.
    // そこから立体角を計算したときに ω = S / r^2
    // r = 1 となる.
    // 今回の場合は、太陽を円盤として立体角は
    //  S = pi x radius ^ 2
    aten::vec3 GetSolarRadiance(const aten::sky::AtmosphereParameters& atmosphere)
    {
        const auto sun_solid_angle = AT_MATH_PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius;
        return atmosphere.solar_irradiance / sun_solid_angle;
    }

    namespace {
        aten::tuple<aten::vec3, aten::vec3> GetCombinedScattering(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::PreComputeTextures texture,
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
            const auto scattering{ scattering_0  * (1.0 - lerp) + scattering_1 * lerp };

            const auto single_mie_scattering_0{ aten::sky::SampleTexture3D(texture.optional_single_mie_scattering_texture, uvw0) };
            const auto single_mie_scattering_1{ aten::sky::SampleTexture3D(texture.optional_single_mie_scattering_texture, uvw1) };
            const auto single_mie_scattering{
                single_mie_scattering_0* (1.0 - lerp) + single_mie_scattering_1 * lerp
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
    aten::vec3 GetSkyRadiance(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& camera,
        const aten::vec3& view_ray,
        Length shadow_length,
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
}
