#pragma once

#include "atmosphere/sky/sky_model.h"
#include "atmosphere/rainbow/rainbow_precompute_textures.h"

#include "camera/camera.h"
#include "math/aabb.h"
#include "image/texture.h"
#include "image/texture_3d.h"
#include "renderer/film.h"

namespace idaten {
    class Atmosphere;
}

namespace aten::rainbow {
    class RainbowModel : public sky::SkyModel {
        friend class idaten::Atmosphere;

    public:
        RainbowModel() = default;
        ~RainbowModel() = default;

        void Init(const aten::CameraParameter& camera);

        void PreCompute();

        void Render(
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

    protected:
        aten::rainbow::PreComputeTextureManager<aten::texture, aten::texture3d> textures_;

        aten::aabb rain_volume_;

        // TODO
        static constexpr float intensity_rainfall_rate = 3.0F; // [mm/h]
    };
}
