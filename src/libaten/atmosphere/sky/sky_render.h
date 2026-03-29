#pragma once

#include "atmosphere/sky/sky_common.h"
#include "atmosphere/sky/sky_compute.h"
#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"

#include "camera/pinhole.h"

#include "math/vec3.h"

#include "misc/tuple.h"

#include "image/texture.h"
#include "image/texture_3d.h"

namespace aten::sky {
    aten::vec3 GetSolarRadiance(const aten::sky::AtmosphereParameters& atmosphere);

    aten::vec3 GetSkyRadiance(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& camera,
        const aten::vec3& view_ray,
        const float shadow_length,
        const aten::vec3& sun_direction,
        aten::vec3& out_transmittance);

    aten::vec3 RenderSky(
        int32_t x, int32_t y,
        const aten::CameraParameter& camera,
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::sky::PreComputeTextures& texture,
        const aten::vec3& sun_radiance_to_luminance,
        const aten::vec3& sky_radiance_to_luminance,
        const aten::vec3& sun_direction,
        const aten::vec3& earth_center,
        const float sun_size);

    aten::vec3 ComputeSpectralRadianceToLuminanceFactors(
        const std::vector<float>& wavelengths,
        const std::vector<float>& solar_irradiance,
        const float lambda_power);

    aten::vec3 ConvertSpectrumToLinearSrgb(
        const std::vector<float>& wavelengths,
        const std::vector<float>& spectrum);
}
