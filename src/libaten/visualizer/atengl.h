#pragma once

#include <GL/glew.h>
#include "defs.h"
#include "visualizer/pixelformat.h"

#define CALL_GL_API(func)\
    func; \
    {\
        GLenum __gl_err__ = ::glGetError();\
        if (__gl_err__ != GL_NO_ERROR) { AT_PRINTF("GL Error[0x%x](%s[%d])\n", __gl_err__, __FILE__, __LINE__); AT_ASSERT(false); }\
    }

namespace aten {
    inline void GetGLPixelFormat(
        PixelFormat fmt,
        GLenum& glfmt,
        GLenum& gltype,
        GLenum& glinternal)
    {
        constexpr GLenum glpixelfmt[] = {
            GL_RGBA,
            GL_RGBA,
            GL_RGBA,
        };

        constexpr GLenum glpixeltype[] = {
            GL_UNSIGNED_BYTE,
            GL_FLOAT,
            GL_HALF_FLOAT,
        };

        constexpr GLenum glpixelinternal[] = {
            GL_RGBA,
            GL_RGBA32F,
            GL_RGBA16F,
        };

        int32_t idx = static_cast<int32_t>(fmt);

        glinternal = glpixelinternal[idx];
        glfmt = glpixelfmt[idx];
        gltype = glpixeltype[idx];
    }
}
