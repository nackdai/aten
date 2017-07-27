#pragma once
#include "visualizer/atengl.h"
#include "visualizer/fbo.h"

namespace aten {
	bool FBO::init(int width, int height, PixelFormat fmt)
	{
		if (m_fbo > 0) {
			// TODO
			// Check size, format...

			return true;
		}

		CALL_GL_API(glGenFramebuffers(1, &m_fbo));

		CALL_GL_API(glGenTextures(1, &m_tex));

		CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_tex));

		GLenum pixelfmt = 0;
		GLenum pixeltype = 0;
		GLenum pixelinternal = 0;

		getGLPixelFormat(
			fmt,
			pixelfmt, pixeltype, pixelinternal);

		CALL_GL_API(glTexImage2D(
			GL_TEXTURE_2D,
			0,
			pixelinternal,
			width, height,
			0,
			pixelfmt,
			pixeltype,
			nullptr));

		m_width = width;
		m_height = height;

		return true;
	}

	void FBO::setAsTexture()
	{
		AT_ASSERT(isValid());

		CALL_GL_API(glBindTexture(
			GL_TEXTURE_2D,
			m_tex));

		// Specify filter after binding!!!!!
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

	}

	void FBO::setFBO()
	{
		AT_ASSERT(isValid());

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

		glFramebufferTexture2D(
			GL_FRAMEBUFFER,
			GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D,
			m_tex,
			0);
	}
}
