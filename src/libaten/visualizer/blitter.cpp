#include "visualizer/atengl.h"
#include "visualizer/blitter.h"

namespace aten {
    void Blitter::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        GLfloat invScreen[4] = { 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f };
        auto hInvScreen = getHandle("invScreen");
        if (hInvScreen >= 0) {
            CALL_GL_API(::glUniform4fv(hInvScreen, 1, invScreen));
        }

        auto hRevert = getHandle("revert");
        if (hRevert >= 0) {
            CALL_GL_API(::glUniform1i(hRevert, revert ? 1 : 0));
        }

        auto hImage = getHandle("image");
        if (hImage >= 0) {
            CALL_GL_API(glUniform1i(hImage, 0));
        }

        auto hIsRenderRGB = getHandle("isRenderRGB");
        if (hIsRenderRGB >= 0) {
            CALL_GL_API(::glUniform1i(hIsRenderRGB, m_isRenderRGB ? 1 : 0));
        }
    }
}