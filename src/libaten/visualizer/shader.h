#pragma once

#include "types.h"

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

		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathGS,
			const char* pathFS);

		virtual void prepareRender(
			const void* pixels,
			bool revert);

		//GLint getHandle(const char* name);
		int getHandle(const char* name);

		uint32_t getProgramHandle() const
		{
			return m_program;
		}

		bool isValid() const
		{
			return (m_program > 0);
		}

	protected:
		//GLuint m_program{ 0 };
		uint32_t m_program{ 0 };
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
	};
}