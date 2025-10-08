#include "visualizer/atengl.h"
#include "display/gamma.h"

namespace aten
{
    void GammaCorrection::PrepareRender(
        const void* pixels,
        bool revert)
    {
        Blitter::PrepareRender(pixels, revert);

        auto hGamma = GetHandle("gamma");
        CALL_GL_API(::glUniform1f(hGamma, m_gamma));
    }
}
