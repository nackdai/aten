#pragma once

#include "atmosphere/rainbow/rainbow_constants.h"

#include "image/texture.h"

namespace aten::rainbow {
    template <class texture2d, class texture3d>
    struct PreComputeTextureManager {
        texture2d transmittance_texture;
        texture3d airy_func_tex;
        texture3d droplet_radius_tex;
        texture2d transmittance_in_rain_volume_texture;

#ifdef __CUDACC__
        template <class T>
        void Init(T& texture_host)
        {
            texture_host.transmittance_texture.Init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                aten::TextureFilterMode::Linear);

            texture_host.transmittance_in_rain_volume_texture.Init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                aten::TextureFilterMode::Linear);

            texture_host.airy_func_tex.Init(
                THETA_WIDTH, WAVELENGTH_WIDTH, A_WIDTH,
                aten::TextureFilterMode::Linear);

            texture_host.droplet_radius_tex.Init(
                DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE,
                aten::TextureFilterMode::Linear);

            transmittance_texture = texture_host.transmittance_texture.GetSurfaceTexture();
            transmittance_in_rain_volume_texture = texture_host.transmittance_in_rain_volume_texture.GetSurfaceTexture();
            airy_func_tex = texture_host.airy_func_tex.GetSurfaceTexture();
            droplet_radius_tex = texture_host.droplet_radius_tex.GetSurfaceTexture();
        }
#else
        void Init()
        {
            transmittance_texture.init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                3);

            transmittance_in_rain_volume_texture.init(
                aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
                aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
                3);

            // Init 3d texture to store Airy function values.
            airy_func_tex.init(THETA_WIDTH, WAVELENGTH_WIDTH, A_WIDTH);

            // Init 3d texture to store droplet radius based on normal distribution.
            droplet_radius_tex.init(DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE, DROPLET_RADIUS_TEX_SIZE);
       }
#endif
    };
}