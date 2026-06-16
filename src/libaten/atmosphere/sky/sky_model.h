#pragma once

#include <vector>

#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/star_types.h"
#include "atmosphere/sky/sky_types.h"

#include "camera/camera.h"
#include "image/texture.h"
#include "image/texture_3d.h"
#include "renderer/film.h"

namespace idaten {
    class Atmosphere;
}

namespace aten::sky {
    class SkyModel {
        friend class idaten::Atmosphere;

    public:
        // TODO
        static constexpr int32_t NUM_SCATTERING = 4;

        SkyModel() = default;
        ~SkyModel() = default;

        aten::sky::AtmosphereParameters& GetMutableAtmoshphereParam()
        {
            return atmosphere_;
        }

        const aten::sky::AtmosphereParameters& GetAtmoshphereParam() const
        {
            return atmosphere_;
        }

        void Init();

        void PreCompute();

        void Render(
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

        void RenderNightSky(
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

        void RenderNightSky(
            const std::vector<aten::sky::Star>& stars,
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

        void RenderStars(
            const std::vector<aten::sky::Star>& stars,
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

    protected:
        static void InitParameters(SkyModel& sky_model);

        aten::sky::AtmosphereParameters atmosphere_;
        aten::mat4 luminance_from_radiance_;

        aten::vec3 sun_radiance_to_luminance_;
        aten::vec3 sky_radiance_to_luminance_;

        aten::vec3 precompute_reference_irradiance_{ 1.0F };
        aten::vec3 sun_light_irradiance_{ 1.0F };

        aten::vec3 white_point_;

    private:
        aten::sky::PreComputeTextureManager<aten::texture, aten::texture3d> textures_;
        std::vector<aten::vec3> hdr_buffer_;
    };
}
