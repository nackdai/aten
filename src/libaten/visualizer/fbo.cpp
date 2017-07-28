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

		m_tex.resize(m_num);

		CALL_GL_API(glGenTextures(m_num, &m_tex[0]));

		for (int i = 0; i < m_num; i++) {
			CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_tex[i]));

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
		}

		m_width = width;
		m_height = height;

		return true;
	}

	void FBO::setAsTexture(uint32_t idx/*= 0*/)
	{
		AT_ASSERT(m_tex[idx] > 0);

		CALL_GL_API(glBindTexture(
			GL_TEXTURE_2D,
			m_tex[idx]));

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

		for (int i = 0; i < m_num; i++) {
			glFramebufferTexture2D(
				GL_FRAMEBUFFER,
				GL_COLOR_ATTACHMENT0 + i,
				GL_TEXTURE_2D,
				m_tex[i],
				0);
		}
	}

	void FBO::asMulti(uint32_t num)
	{
		AT_ASSERT(num > 0);
		AT_ASSERT(m_fbo == 0);

		m_num = num;
	}
}
