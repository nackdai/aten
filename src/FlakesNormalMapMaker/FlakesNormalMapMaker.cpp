#include "visualizer/atengl.h"
#include "FlakesNormalMapMaker.h"

namespace aten {
    void FlakesNormalMapMaker::PrepareRender(
        const void* pixels,
        bool revert)
    {
        shader::PrepareRender(pixels, revert);

        std::array<float, 4> resolution = {
            static_cast<float>(width_),
            static_cast<float>(height_),
            0.0f, 0.0f
        };
        auto hRes = GetHandle("u_resolution");
        if (hRes >= 0) {
            CALL_GL_API(::glUniform4fv(hRes, 1, resolution.data()));
        }

        shader::SetUniformFloat("flake_scale", param_.flake_scale);
        shader::SetUniformFloat("flake_size", param_.flake_size);
        shader::SetUniformFloat("flake_size_variance", param_.flake_size_variance);
        shader::SetUniformFloat("flake_normal_orientation", param_.flake_normal_orientation);
    }
}
