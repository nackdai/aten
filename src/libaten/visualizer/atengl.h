#pragma once

#include <GL/glew.h>
#include "defs.h"
#include "visualizer/pixelformat.h"

#define CALL_GL_API(func)\
    func; \
    {\
        GLenum __gl_err__ = ::glGetError();\
        if (__gl_err__ != GL_NO_ERROR) { AT_PRINTF("GL Error[%x](%s[%d])\n", __gl_err__, __FILE__, __LINE__); AT_ASSERT(false); }\
    }

namespace aten {
    inline void getGLPixelFormat(
        PixelFormat fmt,
        GLenum& glfmt,
        GLenum& gltype,
        GLenum& glinternal)
    {
        static GLenum glpixelfmt[] = {
            GL_RGBA,
            GL_RGBA,
            GL_RGBA,
        };

        static GLenum glpixeltype[] = {
            GL_UNSIGNED_BYTE,
            GL_FLOAT,
            GL_HALF_FLOAT,
        };

        static GLenum glpixelinternal[] = {
            GL_RGBA,
            GL_RGBA32F,
            GL_RGBA16F,
        };

        glinternal = glpixelinternal[fmt];
        glfmt = glpixelfmt[fmt];
        gltype = glpixeltype[fmt];
    }
}
