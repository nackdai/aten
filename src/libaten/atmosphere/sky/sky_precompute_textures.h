#pragma once

#include "atmosphere/sky/sky_constants.h"

#include "image/texture.h"
#include "image/texture_3d.h"

namespace aten::sky {
    struct PreComputeTextures {
        // Permanent.
        aten::texture transmittance_texture;
        aten::texture irradiance_texture;
        aten::texture3d scattering_texture;
        aten::texture3d optional_single_mie_scattering_texture;

        // One shot.
        aten::texture delta_irradiance_texture;
        aten::texture3d delta_rayleigh_scattering_texture;
        aten::texture3d delta_mie_scattering_texture;
        aten::texture3d delta_scattering_density_texture;
        aten::texture3d delta_multiple_scattering_texture;

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
    };
}