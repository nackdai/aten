#pragma once

#include "visualizer/fbo.h"

namespace aten {
	bool fbo::init(int width, int height, PixelFormat fmt)
	{
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
}