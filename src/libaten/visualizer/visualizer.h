#pragma once

#include <GL/glew.h>
#include "defs.h"

namespace aten {
	class visualizer {
		static visualizer s_instance;

	private:
		visualizer() {}
		~visualizer() {}

	public:
		static visualizer& instance()
		{
			return s_instance;
		}

		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathPS);

		void beginRender();

		void endRender(const void* pixels);

	private:
		GLuint createTexture(int width, int height);

		GLuint createShader(const char* path, GLenum type);

		GLuint createProgram(GLuint vs, GLuint fs);

	private:
		GLuint m_program{ 0 };
		GLuint m_tex{ 0 };

		int m_width{ 0 };
		int m_height{ 0 };
	};
}