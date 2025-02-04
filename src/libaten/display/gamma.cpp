#include "visualizer/atengl.h"
#include "display/gamma.h"

namespace aten
{
    void GammaCorrection::prepareRender(
        const void* pixels,
        bool revert)
    {
        Blitter::prepareRender(pixels, revert);

        auto hGamma = getHandle("gamma");
        CALL_GL_API(::glUniform1f(hGamma, m_gamma));
    }
}
