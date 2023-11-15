#include "visualizer/atengl.h"
#include "FlakesNormalMapMaker.h"

namespace aten {
    FlakesNormalMapMaker::Parameter FlakesNormalMapMaker::s_param;

    void FlakesNormalMapMaker::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        float resolution[4] = { width_, height_, 0.0f, 0.0f };
        auto hRes = getHandle("u_resolution");
        if (hRes >= 0) {
            CALL_GL_API(::glUniform4fv(hRes, 1, resolution));
        }

        shader::setUniformFloat("flake_scale", s_param.flake_scale);
        shader::setUniformFloat("flake_size", s_param.flake_size);
        shader::setUniformFloat("flake_size_variance", s_param.flake_size_variance);
        shader::setUniformFloat("flake_normal_orientation", s_param.flake_normal_orientation);
    }
}
