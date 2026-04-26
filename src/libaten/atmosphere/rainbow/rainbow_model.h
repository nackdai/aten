#pragma once

#include "atmosphere/sky/sky_model.h"

#include "camera/camera.h"
#include "image/texture.h"
#include "image/texture_3d.h"
#include "renderer/film.h"

namespace aten::rainbow {
    class RainbowModel : public sky::SkyModel {
    public:
        RainbowModel() = default;
        ~RainbowModel() = default;

        void Init();

        void PreCompute();

        void Render(
            const int32_t width,
            const int32_t height,
            const aten::CameraParameter& camera,
            Film& dst);

    protected:
        aten::texture3d airy_func_tex_;
        aten::texture transmittance_texture_;
    };
}
