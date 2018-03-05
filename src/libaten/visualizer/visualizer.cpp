#include <vector>
#include "visualizer/atengl.h"
#include "visualizer/visualizer.h"
#include "math/vec3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace aten
{
	visualizer* visualizer::s_curVisualizer = nullptr;

	PixelFormat visualizer::getPixelFormat()
	{
		return m_fmt;
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

	uint32_t visualizer::getTexHandle()
	{
		AT_ASSERT(s_curVisualizer);
		return s_curVisualizer->m_tex;
	}

	visualizer* visualizer::init(int width, int height)
	{
		visualizer* ret = new visualizer();

		ret->m_tex = createTexture(width, height, ret->m_fmt);
		AT_VRETURN(ret->m_tex != 0, false);

		ret->m_width = width;
		ret->m_height = height;

		if (!s_curVisualizer) {
			s_curVisualizer = ret;
		}

		return ret;
	}

	void visualizer::addPreProc(PreProc* preproc)
	{
		m_preprocs.push_back(preproc);
	}

	bool visualizer::addPostProc(PostProc* postproc)
	{
		if (m_postprocs.size() > 0) {
			// Create fbo to connect between post-processes.
			auto idx = m_postprocs.size() - 1;
			auto* prevPostproc = m_postprocs[idx];
			auto outFmt = prevPostproc->outFormat();

			// Check in-out format.
			auto inFmt = postproc->inFormat();
			AT_VRETURN(inFmt == outFmt, false);

			auto& fbo = prevPostproc->getFbo();

			// Create FBO.
			AT_VRETURN(fbo.init(m_width, m_height, outFmt), false);
		}

		m_postprocs.push_back(postproc);

		return true;
	}

	static inline GLint getHandle(GLuint program, const char* name)
	{
		auto handle = ::glGetUniformLocation(program, name);
		return handle;
	}

	const void* visualizer::doPreProcs(const vec4* pixels)
	{
		const void* textureimage = pixels;

		if (!m_preprocs.empty()) {
			uint32_t bufpos = 0;
			const vec4* src = (const vec4*)textureimage;
			vec4* dst = nullptr;

			for (int i = 0; i < m_preprocs.size(); i++) {
				auto& buf = m_preprocBuffer[bufpos];
				if (buf.empty()) {
					buf.resize(m_width * m_height);
				}
				dst = &buf[0];

				(*m_preprocs[i])(src, m_width, m_height, dst);

				src = dst;
				bufpos = 1 - bufpos;
			}

			textureimage = src;
		}

		return textureimage;
	}

	const void* visualizer::convertTextureData(const void* textureimage)
	{
		// If type is double, convert double/rgb to float/rgba.
		// If type is float, convert rgb to rgba.
		if (m_tmp.empty()) {
			m_tmp.resize(m_width * m_height);
		}

		const vec4* src = (const vec4*)textureimage;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < m_height; y++) {
			for (int x = 0; x < m_width; x++) {
				int pos = y * m_width + x;

				auto& s = src[pos];
				auto& d = m_tmp[pos];

				d.r() = (float)s.x;
				d.g() = (float)s.y;
				d.b() = (float)s.z;
				d.a() = (float)s.w;
			}
		}

		textureimage = &m_tmp[0];

		return textureimage;
	}

	void visualizer::render(
		const vec4* pixels,
		bool revert)
	{
		s_curVisualizer = this;

		// Do pre processes.
		const void* textureimage = doPreProcs(pixels);

		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, m_tex));

		// Converte texture data double->float, rgb->rgba.
		textureimage = convertTextureData(textureimage);

		GLenum pixelfmt = 0;
		GLenum pixeltype = 0;
		GLenum pixelinternal = 0;

		getGLPixelFormat(
			m_fmt,
			pixelfmt, pixeltype, pixelinternal);

		CALL_GL_API(::glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			0, 0,
			m_width, m_height,
			pixelfmt,
			pixeltype,
			textureimage));

		bool willRevert = revert;

		for (int i = 0; i < m_postprocs.size(); i++) {
			auto* postproc = m_postprocs[i];
			PostProc* prevPostproc = nullptr;

			if (i > 0) {
				prevPostproc = m_postprocs[i - 1];
				auto& fbo = prevPostproc->getFbo();

				CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

				// Set FBO as source texture.
				fbo.bindAsTexture();
			}

			postproc->prepareRender(prevPostproc, pixels, willRevert);
			auto& fbo = postproc->getFbo();

			// ç≈èâÇÃÇPâÒÇæÇØîΩì]Ç∑ÇÍÇŒÇ¢Ç¢ÇÃÇ≈.
			willRevert = false;

			if (fbo.isValid()) {
				// Set FBO.
				fbo.bindFBO();
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
		render(m_tex, revert);
	}

	void visualizer::render(uint32_t gltex, bool revert)
	{
		// For using written OpenGL texture resource by GPGPU directly.
		// So, not handle pixel data pointer directly.

		CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, gltex));

		// TODO
#if 0
		// Specify filter after binding!!!!!
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
#endif

		bool willRevert = revert;

		for (int i = 0; i < m_postprocs.size(); i++) {
			auto* postproc = m_postprocs[i];
			PostProc* prevPostproc = nullptr;

			if (i > 0) {
				prevPostproc = m_postprocs[i - 1];
				auto& fbo = prevPostproc->getFbo();

				CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

				// Set FBO as source texture.
				fbo.bindAsTexture();
			}

			postproc->prepareRender(prevPostproc, nullptr, willRevert);
			auto& fbo = postproc->getFbo();

			// ç≈èâÇÃÇPâÒÇæÇØîΩì]Ç∑ÇÍÇŒÇ¢Ç¢ÇÃÇ≈.
			willRevert = false;

			if (fbo.isValid()) {
				// Set FBO.
				fbo.bindFBO();
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
			m_fmt,
			pixelfmt, pixeltype, pixelinternal);

		const void* clearclr = nullptr;

		switch (m_fmt) {
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
			m_tex,
			0,
			pixelfmt, pixeltype,
			clearclr));
	}

	void visualizer::takeScreenshot(const char* filename)
	{
		CALL_GL_API(::glFlush());
		CALL_GL_API(::glFinish());

		using ScreenShotImageType = TColor<uint8_t, 3>;

		std::vector<ScreenShotImageType> tmp(m_width * m_height);

		CALL_GL_API(::glBindFramebuffer(GL_READ_FRAMEBUFFER, 0));
		CALL_GL_API(::glNamedFramebufferReadBuffer(0, GL_BACK));

		CALL_GL_API(::glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, &tmp[0]));

		// up-side-down.
		std::vector<ScreenShotImageType> dst(m_width * m_height);

		static const int bpp = sizeof(ScreenShotImageType);
		const int pitch = m_width * bpp;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < m_height; y++) {
			int yy = m_height - 1 - y;

			memcpy(
				&dst[yy * m_width],
				&tmp[y * m_width],
				pitch);
		}

		auto ret = ::stbi_write_png(filename, m_width, m_height, bpp, &dst[0], pitch);
		AT_ASSERT(ret > 0);
	}

	void visualizer::getTextureData(
		uint32_t gltex, 
		std::vector<TColor<uint8_t, 4>>& dst)
	{
		CALL_GL_API(::glFlush());
		CALL_GL_API(::glFinish());

		auto size = dst.size() * sizeof(uint8_t) * 4;

		CALL_GL_API(::glGetTextureImage(
			gltex,
			0,
			GL_RGBA,
			GL_UNSIGNED_BYTE,
			size,
			&dst[0]));
	}
}