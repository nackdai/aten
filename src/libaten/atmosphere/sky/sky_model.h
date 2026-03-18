#pragma once

#include "atmosphere/sky/sky_params.h"
#include "atmosphere/sky/sky_precompute_textures.h"

namespace AT_NAME::sky {
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

        void Render();

    private:
        // TODO
        static constexpr int32_t NUM_SCATTERING = 4;

        aten::sky::AtmosphereParameters atmosphere_;
        aten::sky::SceneParameters scene_;
        aten::sky::PreComputeTextures textures_;
    };
}
