#pragma once

#include "types.h"
#include "visualizer/atengl.h"

namespace aten {
	class fbo {
	public:
		fbo() {}
		virtual ~fbo() {}

	public:
		bool init(int width, int height, PixelFormat fmt);

	protected:
		GLuint m_fbo{ 0 };
		GLuint m_tex{ 0 };
		PixelFormat m_fmt{ PixelFormat::rgba8 };
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
	};
}