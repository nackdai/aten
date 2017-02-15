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
	class shader {
	public:
		shader() {}
		virtual ~shader() {}

	public:
		virtual bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		virtual void begin(const void* pixels);

		GLint getHandle(const char* name);

	protected:
		GLuint m_program{ 0 };
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
	};

	class SimpleRender : public shader {
	public:
		SimpleRender() {}
		virtual ~SimpleRender() {}

	public:
		virtual void begin(const void* pixels) override;
	};
}