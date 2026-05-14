#pragma once

#include "atmosphere/sky/sky_model.h"

#include "camera/camera.h"
#include "math/aabb.h"
#include "image/texture.h"
#include "image/texture_3d.h"
#include "renderer/film.h"

namespace aten::rainbow {
    class RainbowModel : public sky::SkyModel {
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
        aten::texture3d airy_func_tex_;
        aten::texture3d droplet_radius_tex_;
        aten::texture transmittance_texture_;
        aten::texture transmittance_in_rain_volume_texture_;

        aten::aabb rain_volume_;

        // TODO
        static constexpr float intensity_rainfall_rate = 1.0F; // [mm/h]
    };
}
