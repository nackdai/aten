#include <vector>
#include "visualizer/atengl.h"
#include "visualizer/visualizer.h"
#include "math/vec3.h"
#include "misc/color.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace aten {
	static GLuint g_tex{ 0 };

	static int g_width{ 0 };
	static int g_height{ 0 };

	static std::vector<TColor<float, 4>> g_tmp;

	static const PixelFormat g_fmt{ PixelFormat::rgba32f };

	static std::vector<visualizer::PreProc*> g_preprocs;
	static std::vector<vec4> g_preprocBuffer[2];

	static std::vector<visualizer::PostProc*> g_postprocs;

	GLuint visualizer::getTexHandle()
	{
		return g_tex;
	}

	PixelFormat visualizer::getPixelFormat()
	{
		return g_fmt;
	}

	static GLuint createTexture(int width, int height, PixelFormat fmt)
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

	bool visualizer::init(int width, int height)
	{
		g_tex = createTexture(width, height, g_fmt);
		AT_VRETURN(g_tex != 0, false);

		g_width = width;
		g_height = height;

		return true;
	}

	void visualizer::addPreProc(PreProc* preproc)
	{
		g_preprocs.push_back(preproc);
	}

	bool visualizer::addPostProc(PostProc* postproc)
	{
		if (g_postprocs.size() > 0) {
			// Create fbo to connect between post-processes.
			auto idx = g_postprocs.size() - 1;
			auto* prevPostproc = g_postprocs[idx];
			auto outFmt = prevPostproc->outFormat();

			// Check in-out format.
			auto inFmt = postproc->inFormat();
			AT_VRETURN(inFmt == outFmt, false);

			auto& fbo = prevPostproc->getFbo();

			// Create FBO.
			AT_VRETURN(fbo.init(g_width, g_height, outFmt), false);
		}

		g_postprocs.push_back(postproc);

		return true;
	}

	static inline GLint getHandle(GLuint program, const char* name)
	{
		auto handle = ::glGetUniformLocation(program, name);
		return handle;
	}

	static const void* doPreProcs(const vec4* pixels)
	{
		const void* textureimage = pixels;

		if (!g_preprocs.empty()) {
			uint32_t bufpos = 0;
			const vec4* src = (const vec4*)textureimage;
			vec4* dst = nullptr;

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

		return textureimage;
	}

	static const void* convertTextureData(const void* textureimage)
	{
		// If type is double, convert double/rgb to float/rgba.
		// If type is float, convert rgb to rgba.
		if (g_tmp.empty()) {
			g_tmp.resize(g_width * g_height);
		}

		const vec4* src = (const vec4*)textureimage;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < g_height; y++) {
			for (int x = 0; x < g_width; x++) {
				int pos = y * g_width + x;

				auto& s = src[pos];
				auto& d = g_tmp[pos];

				d.r() = (float)s.x;
				d.g() = (float)s.y;
				d.b() = (float)s.z;
				d.a() = (float)s.w;
			}
		}

		textureimage = &g_tmp[0];

		return textureimage;
	}

	void visualizer::render(
		const vec4* pixels,
		bool revert)
	{
		// Do pre processes.
		const void* textureimage = doPreProcs(pixels);

		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, g_tex));

		// Converte texture data double->float, rgb->rgba.
		textureimage = convertTextureData(textureimage);

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

		bool willRevert = revert;

		for (int i = 0; i < g_postprocs.size(); i++) {
			auto* postproc = g_postprocs[i];
			PostProc* prevPostproc = nullptr;

			if (i > 0) {
				prevPostproc = g_postprocs[i - 1];
				auto& fbo = prevPostproc->getFbo();

				CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

				// Set FBO as source texture.
				fbo.setAsTexture();
			}

			postproc->prepareRender(prevPostproc, pixels, willRevert);
			auto& fbo = postproc->getFbo();

			// ç≈èâÇÃÇPâÒÇæÇØîΩì]Ç∑ÇÍÇŒÇ¢Ç¢ÇÃÇ≈.
			willRevert = false;

			if (fbo.isValid()) {
				// Set FBO.
				fbo.setFBO();
			}
			else {
				// Set default frame buffer.
				CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
			}

			CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
		}
	}

	void visualizer::render(bool revert)
	{
		render(g_tex, revert);
	}

	void visualizer::render(uint32_t gltex, bool revert)
	{
		// For using written OpenGL texture resource by GPGPU directly.
		// So, not handle pixel data pointer directly.

		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, gltex));

		bool willRevert = revert;

		for (int i = 0; i < g_postprocs.size(); i++) {
			auto* postproc = g_postprocs[i];
			PostProc* prevPostproc = nullptr;

			if (i > 0) {
				prevPostproc = g_postprocs[i - 1];
				auto& fbo = prevPostproc->getFbo();

				CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

				// Set FBO as source texture.
				fbo.setAsTexture();
			}

			postproc->prepareRender(prevPostproc, nullptr, willRevert);
			auto& fbo = postproc->getFbo();

			// ç≈èâÇÃÇPâÒÇæÇØîΩì]Ç∑ÇÍÇŒÇ¢Ç¢ÇÃÇ≈.
			willRevert = false;

			if (fbo.isValid()) {
				// Set FBO.
				fbo.setFBO();
			}
			else {
				// Set default frame buffer.
				CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
			}

			CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
		}
	}

	void visualizer::clear()
	{
		static const uint8_t clearclr_uc[4] = { 0, 0, 0, 0 };
		static const uint16_t clearclr_us[4] = { 0, 0, 0, 0 };
		static const uint32_t clearclr_ui[4] = { 0, 0, 0, 0 };

		GLenum pixelfmt = 0;
		GLenum pixeltype = 0;
		GLenum pixelinternal = 0;

		getGLPixelFormat(
			g_fmt,
			pixelfmt, pixeltype, pixelinternal);

		const void* clearclr = nullptr;

		switch (g_fmt) {
		case rgba8:
			clearclr = clearclr_uc;
			break;
		case rgba32f:
			clearclr = clearclr_ui;
			break;
		case rgba16f:
			clearclr = clearclr_us;
			break;
		}

		CALL_GL_API(::glClearTexImage(
			g_tex,
			0,
			pixelfmt, pixeltype,
			clearclr));
	}

	void visualizer::takeScreenshot(const char* filename)
	{
		CALL_GL_API(::glFlush());
		CALL_GL_API(::glFinish());

		using ScreenShotImageType = TColor<uint8_t, 3>;

		std::vector<ScreenShotImageType> tmp(g_width * g_height);

		CALL_GL_API(::glReadBuffer(GL_BACK));

		CALL_GL_API(::glReadPixels(0, 0, g_width, g_height, GL_RGB, GL_UNSIGNED_BYTE, &tmp[0]));

		// up-side-down.
		std::vector<ScreenShotImageType> dst(g_width * g_height);

		static const int bpp = sizeof(ScreenShotImageType);
		const int pitch = g_width * bpp;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < g_height; y++) {
			int yy = g_height - 1 - y;

			memcpy(
				&dst[yy * g_width],
				&tmp[y * g_width],
				pitch);
		}

		auto ret = ::stbi_write_png(filename, g_width, g_height, bpp, &dst[0], pitch);
		AT_ASSERT(ret > 0);
	}
}