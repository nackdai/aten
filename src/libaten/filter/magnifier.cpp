#include "filter/magnifier.h"
#include "visualizer/atengl.h"

namespace aten {
    void Magnifier::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        auto handle = getHandle("image");
        if (handle >= 0) {
            CALL_GL_API(glUniform1i(handle, 0));
        }

        handle = getHandle("screen_res");
        if (handle >= 0) {
            CALL_GL_API(glUniform2fv(handle, 1, reinterpret_cast<const GLfloat*>(&screen_res_)));
        }

        handle = getHandle("center_pos");
        if (handle >= 0) {
            CALL_GL_API(glUniform2fv(handle, 1, reinterpret_cast<const GLfloat*>(&center_pos_)));
        }

        handle = getHandle("magnification");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, magnification_));
        }

        handle = getHandle("radius");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, radius_));
        }

        handle = getHandle("circle_line_width");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, circle_line_width_));
        }

        handle = getHandle("circle_line_color");
        if (handle >= 0) {
            CALL_GL_API(glUniform3fv(handle, 1, reinterpret_cast<const GLfloat*>(&circle_line_color_)));
        }
    }
}
