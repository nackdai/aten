#include "visualizer/atengl.h"
#include "FlakesNormalMapMaker.h"

namespace aten {
    void FlakesNormalMapMaker::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        std::array<float, 4> resolution = {
            static_cast<float>(width_),
            static_cast<float>(height_),
            0.0f, 0.0f
        };
        auto hRes = getHandle("u_resolution");
        if (hRes >= 0) {
            CALL_GL_API(::glUniform4fv(hRes, 1, resolution.data()));
        }

        shader::setUniformFloat("flake_scale", param_.flake_scale);
        shader::setUniformFloat("flake_size", param_.flake_size);
        shader::setUniformFloat("flake_size_variance", param_.flake_size_variance);
        shader::setUniformFloat("flake_normal_orientation", param_.flake_normal_orientation);
    }
}
