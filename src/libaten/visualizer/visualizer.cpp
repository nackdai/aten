#include <vector>
#include "visualizer.h"
#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/fbo.h"

namespace aten {
	static shader* g_shader{ nullptr };
	static GLuint g_tex{ 0 };

	static int g_width{ 0 };
	static int g_height{ 0 };

	static std::vector<TColor<float>> g_tmp;

	static PixelFormat g_fmt{ PixelFormat::rgba8 };

	static std::vector<visualizer::PreProc*> g_preprocs;
	static std::vector<vec3> g_preprocBuffer[2];

	static std::vector<visualizer::PostProc*> g_postprocs;
	static std::vector<fbo> g_fbos;

	GLuint createTexture(int width, int height, PixelFormat fmt)
	{
		GLuint tex = 0;

		CALL_GL_API(::glGenTextures(1, &tex));
		AT_ASSERT(tex != 0);

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, tex));

		GLenum pixelfmt = 0;
		GLenum pixeltype = 0;
		GLenum pixelinternal = 0;

		getGLPixelFormat(
			fmt,
			pixelfmt, pixeltype, pixelinternal);

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

	void visualizer::addPreProc(PreProc* preproc)
	{
		g_preprocs.push_back(preproc);
	}

	void visualizer::addPostProc(PostProc* postproc)
	{
		if (g_postprocs.size() > 0) {
			// Create fbo to connect between post-processes.
			auto idx = g_postprocs.size() - 1;
			const auto* postproc = g_postprocs[idx];
			auto fmt = postproc->outFormat();

			fbo fbo;
			fbo.init(g_width, g_height, fmt);

			g_fbos.push_back(fbo);
		}

		g_postprocs.push_back(postproc);
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
		const vec3* pixels,
		bool revert)
	{
		const void* textureimage = pixels;

		if (!g_preprocs.empty()) {
			uint32_t bufpos = 0;
			const vec3* src = (const vec3*)textureimage;
			vec3* dst = nullptr;

			for (int i = 0; i < g_preprocs.size(); i++) {
				auto& buf = g_preprocBuffer[bufpos];
				if (buf.empty()) {
					buf.resize(g_width * g_height);
				}
				dst = &buf[0];

				(*g_preprocs[i])(src, g_width, g_height, dst);

				src = dst;
				bufpos = 1 - bufpos;
			}

			textureimage = src;
		}

		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, g_tex));

		if (g_fmt == PixelFormat::rgba32f) {
			// If type is double, convert double/rgb to float/rgba.
			// If type is float, convert rgb to rgba.
			if (g_tmp.empty()) {
				g_tmp.resize(g_width * g_height);
			}

			const vec3* src = (const vec3*)textureimage;

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

		GLenum pixelfmt = 0;
		GLenum pixeltype = 0;
		GLenum pixelinternal = 0;

		getGLPixelFormat(
			g_fmt,
			pixelfmt, pixeltype, pixelinternal);

		CALL_GL_API(::glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			0, 0,
			g_width, g_height,
			pixelfmt,
			pixeltype,
			textureimage));

#if 1
		g_shader->prepareRender(pixels, revert);

		CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
#else
		for (int i = 0; i < g_postprocs.size(); i++) {
			auto* postproc = g_postprocs[i];

			postproc->prepareRender(pixels, revert);

			CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
		}
#endif
	}
}