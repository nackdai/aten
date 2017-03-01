#pragma once

#include "types.h"
#include "visualizer/atengl.h"

namespace aten {
	class FBO {
	public:
		FBO() {}
		virtual ~FBO() {}

	public:
		bool init(int width, int height, PixelFormat fmt);

		bool isValid() const
		{
			return (m_fbo > 0 && m_tex > 0);
		}

		void setAsTexture();

		void setFBO();

	protected:
		GLuint m_fbo{ 0 };
		GLuint m_tex{ 0 };
		PixelFormat m_fmt{ PixelFormat::rgba8 };
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
	};
}