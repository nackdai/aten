#pragma once

#include <GL/glew.h>
#include "defs.h"

#define CALL_GL_API(func)\
    func; \
    {\
        GLenum __gl_err__ = ::glGetError();\
        if (__gl_err__ != GL_NO_ERROR) { aten::OutputDebugString("GL Error[%x](%s[%d])\n", __gl_err__, __FILE__, __LINE__); AT_ASSERT(false); }\
    }

namespace aten {
	enum PixelFormat {
		rgba8,
		rgba32f
	};

	inline void getGLPixelFormat(
		PixelFormat fmt,
		GLenum& glfmt,
		GLenum& gltype,
		GLenum& glinternal)
	{
		static GLenum glpixelfmt[] = {
			GL_RGBA,
			GL_RGBA,
		};

		static GLenum glpixeltype[] = {
			GL_UNSIGNED_BYTE,
			GL_FLOAT,
		};

		static GLenum glpixelinternal[] = {
			GL_RGBA,
			GL_RGBA32F,
		};

		glinternal = glpixelinternal[fmt];
		glfmt = glpixelfmt[fmt];
		gltype = glpixeltype[fmt];
	}
}