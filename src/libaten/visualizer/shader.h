#pragma once

#include "visualizer/atengl.h"

namespace aten {
	class shader {
	public:
		shader() {}
		virtual ~shader() {}

	public:
		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		virtual void prepareRender(
			const void* pixels,
			bool revert);

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
		virtual void prepareRender(
			const void* pixels,
			bool revert) override;
	};
}