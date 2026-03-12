#pragma once

#include "atmosphere/sky/sky_constants.h"

#include "image/texture.h"

namespace aten::sky {
    template <class texture2d, class texture3d>
    struct PreComputeTextureManager {
        // Permanent.
        texture2d transmittance_texture;
        texture2d irradiance_texture;
        texture3d scattering_texture;
        texture3d optional_single_mie_scattering_texture;

        // One shot.
        texture2d delta_irradiance_texture;
        texture3d delta_rayleigh_scattering_texture;
        texture3d delta_mie_scattering_texture;
        texture3d delta_scattering_density_texture;
        texture3d delta_multiple_scattering_texture;

#ifdef __CUDACC__
        template <class T>
        void Init(T& texture_host)
        {
            texture_host.transmittance_texture.Init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                aten::TextureFilterMode::Linear);
            texture_host.irradiance_texture.Init(
                aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
                aten::TextureFilterMode::Linear);
            texture_host.scattering_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);
            texture_host.optional_single_mie_scattering_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);

            texture_host.delta_irradiance_texture.Init(
                aten::sky::IRRADIANCE_TEXTURE_WIDTH,
                aten::sky::IRRADIANCE_TEXTURE_HEIGHT,
                aten::TextureFilterMode::Linear);
            texture_host.delta_rayleigh_scattering_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);
            texture_host.delta_mie_scattering_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);
            texture_host.delta_scattering_density_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);
            texture_host.delta_multiple_scattering_texture.Init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH,
                aten::TextureFilterMode::Linear);

            transmittance_texture = texture_host.transmittance_texture.GetSurfaceTexture();
            irradiance_texture = texture_host.irradiance_texture.GetSurfaceTexture();
            scattering_texture = texture_host.scattering_texture.GetSurfaceTexture();
            optional_single_mie_scattering_texture = texture_host.optional_single_mie_scattering_texture.GetSurfaceTexture();

            delta_irradiance_texture = texture_host.delta_irradiance_texture.GetSurfaceTexture();
            delta_rayleigh_scattering_texture = texture_host.delta_rayleigh_scattering_texture.GetSurfaceTexture();
            delta_mie_scattering_texture = texture_host.delta_mie_scattering_texture.GetSurfaceTexture();
            delta_scattering_density_texture = texture_host.delta_scattering_density_texture.GetSurfaceTexture();
            delta_multiple_scattering_texture = texture_host.delta_multiple_scattering_texture.GetSurfaceTexture();
        }
#else
        void Init()
        {
            transmittance_texture.init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                3);
            irradiance_texture.init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                3);
            scattering_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);
            optional_single_mie_scattering_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);

            delta_irradiance_texture.init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                3);
            delta_rayleigh_scattering_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);
            delta_mie_scattering_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);
            delta_scattering_density_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);
            delta_multiple_scattering_texture.init(
                aten::sky::SCATTERING_TEXTURE_WIDTH,
                aten::sky::SCATTERING_TEXTURE_HEIGHT,
                aten::sky::SCATTERING_TEXTURE_DEPTH);
        }
#endif
    };
}