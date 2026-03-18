#include "atmosphere/sky/sky_model.h"

#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_coord_convert.h"

namespace aten::sky {
    namespace {
        // Transmittance を計算.
        void ComputeTransmittanceToTopAtmosphereBoundaryTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures)
        {
            static const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
            };

            const aten::vec2 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
            };

            float r;
            float mu;

            aten::sky::GetRMuFromTransmittanceTextureUv(
                atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);

            // rとmuから上端境界までのtransmittanceを計算する.
            const auto transmittance{
                aten::sky::transmittance::ComputeTransmittanceToTopAtmosphereBoundary(
                    atmosphere,
                    r, mu)
            };

            textures.transmittance_texture.PutByXYcoord(x, y, transmittance);
        }

        static const aten::vec2 IRRADIANCE_TEXTURE_SIZE{
            aten::sky::IRRADIANCE_TEXTURE_WIDTH,
            aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
        };

        // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
        void ComputeDirectIrradianceTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures)
        {
            const aten::vec2 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
            };

            float r;
            float mu_s;

            aten::sky::GetRMuSFromIrradianceTextureUv(
                atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);

            aten::vec3 irradiance{
                ComputeDirectIrradiance(atmosphere, textures.transmittance_texture, r, mu_s)
            };

            textures.delta_irradiance_texture.PutByXYcoord(x, y, irradiance);
            textures.irradiance_texture.PutByXYcoord(x, y, aten::vec3(0.0F));
        }

        // 論文内の 4. Precomputations の Angular precision の部分で説明されている、太陽光からの単一散乱のテクスチャを計算するためのシェーダ.
        //  delta_rayleigh: レイリー散乱の係数のみ. luminanceとの乗算はしていない.
        //  delta_mie: ミー散乱の係数のみ. luminanceとの乗算はしていない.
        //  scattering: 3Dテクスチャ. vec4.rgb に C*（レイリー散乱）, vec4.a に CM.r（ミー散乱） の値が入るテクスチャ. layer 回レンダリングすることで、3次元に値を格納する.
        //  single_mie_scattering: CM （ミー散乱）のみを格納するテクスチャ.
        void ComputeSingleScatteringTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::SceneParameters& scene,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures)
        {
            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            const aten::vec3 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(layer),
            };

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            aten::vec3 rayleigh;
            aten::vec3 mie;

            aten::sky::single_scattering::ComputeSingleScattering(
                atmosphere,
                textures.transmittance_texture,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground,
                rayleigh, mie);

            textures.delta_rayleigh_scattering_texture.SetByXYZ(rayleigh, x, y, layer);
            textures.delta_mie_scattering_texture.SetByXYZ(mie, x, y, layer);

            rayleigh = scene.luminance_from_radiance * rayleigh;
            mie = scene.luminance_from_radiance * mie;

            textures.scattering_texture.SetByXYZ(
                aten::vec4(rayleigh.r, rayleigh.g, rayleigh.b, mie.r),
                x, y, layer);
            textures.optional_single_mie_scattering_texture.SetByXYZ(mie, x, y, layer);
        }

        // ΔJ を計算する.
        void ComputeScatteringDensityTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::SceneParameters& scene,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures,
            const int32_t scattering_order)
        {
            const aten::vec3 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(layer),
            };

            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            const auto scattering_density{
                aten::sky::ComputeScatteringDensity(
                    atmosphere,
                    textures.transmittance_texture,
                    textures.delta_rayleigh_scattering_texture,
                    textures.delta_mie_scattering_texture,
                    textures.delta_multiple_scattering_texture,
                    textures.delta_irradiance_texture,
                    r, mu, mu_s, nu,
                    scattering_order)
            };

            textures.delta_scattering_density_texture.SetByXYZ(
                scattering_density,
                x, y, layer);
        }

        // E = E + ΔE を計算するための、太陽光でない入射する放射照度 ΔE を計算する.
        void ComputeIndirectIrradianceTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::SceneParameters& scene,
            const int32_t x, const int32_t y,
            aten::sky::PreComputeTextures& textures,
            const int32_t scattering_order)
        {
            const aten::vec2 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
            };

            float r;
            float mu_s;
            aten::sky::GetRMuSFromIrradianceTextureUv(
                atmosphere,
                frag_coord / IRRADIANCE_TEXTURE_SIZE,
                r, mu_s);

            auto delta_irradiance{
                ComputeIndirectIrradiance(atmosphere,
                    textures.delta_rayleigh_scattering_texture,
                    textures.delta_mie_scattering_texture,
                    textures.delta_multiple_scattering_texture,
                    r, mu_s,
                    scattering_order)
            };

            const auto curr_irradiance{ textures.irradiance_texture.AtByXY(x, y) };
            delta_irradiance = scene.luminance_from_radiance * delta_irradiance;

            textures.irradiance_texture.PutByXYcoord(
                x, y,
                curr_irradiance + delta_irradiance);
        }

        // S = S + ΔS を計算するための ΔS を計算する.
        void ComputeMultipleScatteringTexture(
            const aten::sky::AtmosphereParameters& atmosphere,
            const aten::sky::SceneParameters& scene,
            const int32_t x, const int32_t y, const int32_t layer,
            aten::sky::PreComputeTextures& textures)
        {
            const aten::vec3 frag_coord{
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(layer),
            };

            float r;
            float mu;
            float mu_s;
            float nu;
            bool ray_r_mu_intersects_ground;

            aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(
                atmosphere,
                frag_coord,
                r, mu, mu_s, nu, ray_r_mu_intersects_ground);

            const auto delta_multiple_scattering{
                ComputeMultipleScattering(
                    atmosphere,
                    textures.transmittance_texture,
                    textures.delta_scattering_density_texture,
                    r, mu, mu_s, nu,
                    ray_r_mu_intersects_ground)
            };

            const auto curr_scattering{ textures.scattering_texture.AtByXYZ(x, y, layer) };

            textures.delta_multiple_scattering_texture.SetByXYZ(
                delta_multiple_scattering,
                x, y, layer);

            const auto phase = aten::sky::RayleighPhaseFunction(nu);

            // C∗ = S_R[L0] + S[L∗]/P_R として保存.
            // P_R は後で乗算して、P_R・S_R[L0]+S[L∗] として計算するため.
            textures.scattering_texture.SetByXYZ(
                curr_scattering + delta_multiple_scattering / phase,
                x, y, layer);
        }
    }

    void SkyModel::Init()
    {
        textures_.Init();
    }

    void SkyModel::PreCompute()
    {
        {
            // Transmittance を計算.
            for (int32_t y = 0; y < aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::TRANSMITTANCE_TEXTURE_WIDTH; x++) {
                    ComputeTransmittanceToTopAtmosphereBoundaryTexture(
                        atmosphere_,
                        x, y,
                        textures_);
                }
            }

            // 最初の ΔE を計算.
            // 太陽からの入射放射輝度から指定された点での放射照度を計算する.
            for (int32_t y = 0; y < aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::TRANSMITTANCE_TEXTURE_WIDTH; x++) {
                    ComputeDirectIrradianceTexture(
                        atmosphere_,
                        x, y,
                        textures_);
                }
            }

            // 最初の ΔS を計算.
            // 太陽光（一方向）からの単一散乱.
            for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                    for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                        ComputeSingleScatteringTexture(
                            atmosphere_,
                            scene_,
                            x, y, z,
                            textures_);
                    }
                }
            }

            // ここまでで、1st scattering order は計算済み。次に、2nd scattering order 以降を順番に計算していく.
            for (int32_t scattering_order = 2; scattering_order <= NUM_SCATTERING; scattering_order++)
            {
                // ΔJ を計算.
                for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                        for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                            ComputeScatteringDensityTexture(
                                atmosphere_,
                                scene_,
                                x, y, z,
                                textures_,
                                scattering_order);
                        }
                    }
                }

                // ΔE を計算して、E = E + ΔE する.
                for (int32_t y = 0; y < aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::TRANSMITTANCE_TEXTURE_WIDTH; x++) {
                        ComputeIndirectIrradianceTexture(
                            atmosphere_,
                            scene_,
                            x, y,
                            textures_,
                            scattering_order - 1);
                    }
                }

                // ΔS を計算して、S = S + ΔS する.
                for (int32_t y = 0; y < aten::sky::SCATTERING_TEXTURE_HEIGHT; y++) {
                    for (int32_t x = 0; x < aten::sky::SCATTERING_TEXTURE_WIDTH; x++) {
                        for (int32_t z = 0; z < aten::sky::SCATTERING_TEXTURE_DEPTH; z++) {
                            ComputeMultipleScatteringTexture(
                                atmosphere_,
                                scene_,
                                x, y, z,
                                textures_);
                        }
                    }
                }
            }
        }
    }
}