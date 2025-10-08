#include "visualizer/atengl.h"
#include "visualizer/blitter.h"

namespace aten {
    void Blitter::PrepareRender(
        const void* pixels,
        bool revert)
    {
        shader::PrepareRender(pixels, revert);

        GLfloat invScreen[4] = { 1.0f / width_, 1.0f / height_, 0.0f, 0.0f };
        auto hInvScreen = GetHandle("invScreen");
        if (hInvScreen >= 0) {
            CALL_GL_API(::glUniform4fv(hInvScreen, 1, invScreen));
        }

        auto hRevert = GetHandle("revert");
        if (hRevert >= 0) {
            CALL_GL_API(::glUniform1i(hRevert, revert ? 1 : 0));
        }

        auto hImage = GetHandle("image");
        if (hImage >= 0) {
            CALL_GL_API(glUniform1i(hImage, 0));
        }
    }
}
