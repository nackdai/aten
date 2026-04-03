#pragma once

#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"

#include "camera/camera.h"
#include "image/texture.h"
#include "image/texture_3d.h"
#include "renderer/film.h"

namespace aten::sky {
    class SkyModel {
    public:
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

    protected:
        void InitParameters();

        // TODO
        static constexpr int32_t NUM_SCATTERING = 4;

        aten::sky::AtmosphereParameters atmosphere_;
        aten::mat4 luminance_from_radiance_;

        aten::vec3 sun_radiance_to_luminance_;
        aten::vec3 sky_radiance_to_luminance_;

        aten::vec3 white_point_;

    private:
        aten::sky::PreComputeTextureManager<aten::texture, aten::texture3d> textures_;
    };
}
