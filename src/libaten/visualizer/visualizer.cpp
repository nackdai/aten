#include <vector>
#include "visualizer.h"
#include "math/vec3.h"
#include "misc/color.h"

namespace aten {
	static shader* g_shader{ nullptr };
	static GLuint g_tex{ 0 };

	static int g_width{ 0 };
	static int g_height{ 0 };

	static std::vector<TColor<float>> g_tmp;

	static PixelFormat g_fmt{ PixelFormat::rgba8 };

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

	GLuint createTexture(int width, int height, PixelFormat fmt)
	{
		GLuint tex = 0;

		CALL_GL_API(::glGenTextures(1, &tex));
		AT_ASSERT(tex != 0);

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, tex));

		auto pixelinternal = glpixelinternal[fmt];
		auto pixelfmt = glpixelfmt[fmt];
		auto pixeltype = glpixeltype[fmt];

		CALL_GL_API(::glTexImage2D(
			GL_TEXTURE_2D,
			0,
			pixelinternal,
			width, height,
			0,
			pixelfmt,
			pixeltype,
			NULL));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));

		return tex;
	}

	bool visualizer::init(int width, int height, PixelFormat fmt)
	{
		GLenum result = glewInit();
		AT_ASSERT(result == GLEW_OK);

		auto version = ::glGetString(GL_VERSION);
		AT_PRINTF("GL Version(%s)\n", version);

		CALL_GL_API(::glClipControl(
			GL_LOWER_LEFT,
			GL_ZERO_TO_ONE));

		CALL_GL_API(::glFrontFace(GL_CCW));

		CALL_GL_API(::glViewport(0, 0, width, height));
		CALL_GL_API(::glDepthRangef(0.0f, 1.0f));

		g_tex = createTexture(width, height, fmt);
		AT_VRETURN(g_tex != 0, false);

		g_width = width;
		g_height = height;

		g_fmt = fmt;

		return true;
	}

	void visualizer::setShader(shader* shader)
	{
		g_shader = shader;
	}

	static inline GLint getHandle(GLuint program, const char* name)
	{
		auto handle = ::glGetUniformLocation(program, name);
		return handle;
	}

	void visualizer::render(
		const void* pixels,
		bool revert)
	{
		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, g_tex));

		g_shader->begin(pixels, revert);

		const void* textureimage = pixels;

		if (g_fmt == PixelFormat::rgba32f) {
			// If type is double, convert double/rgb to float/rgba.
			// If type is float, convert rgb to rgba.
			if (g_tmp.empty()) {
				g_tmp.resize(g_width * g_height);
			}

			const vec3* src = (const vec3*)pixels;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
			for (int y = 0; y < g_height; y++) {
				for (int x = 0; x < g_width; x++) {
					int pos = y * g_width + x;

					auto& s = src[pos];
					auto& d = g_tmp[pos];

					d.r = (float)s.x;
					d.g = (float)s.y;
					d.b = (float)s.z;
				}
			}

			textureimage = &g_tmp[0];
		}

		auto pixelfmt = glpixelfmt[g_fmt];
		auto pixeltype = glpixeltype[g_fmt];

		CALL_GL_API(::glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			0, 0,
			g_width, g_height,
			pixelfmt,
			pixeltype,
			textureimage));

		CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
	}
}