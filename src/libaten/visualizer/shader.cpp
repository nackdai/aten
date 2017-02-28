#include <vector>
#include "visualizer/shader.h"

namespace aten {
	GLuint createShader(const char* path, GLenum type)
	{
		FILE* fp = nullptr;
		fopen_s(&fp, path, "rt");
		AT_ASSERT(fp != nullptr);

		fseek(fp, 0, SEEK_END);
		auto size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		std::vector<char> program(size + 1);
		fread(&program[0], 1, size, fp);

		fclose(fp);

		CALL_GL_API(auto shader = ::glCreateShader(type));
		AT_ASSERT(shader != 0);

		const char* p = &program[0];
		const char** pp = &p;

		CALL_GL_API(::glShaderSource(
			shader,
			1,
			pp,
			nullptr));

		CALL_GL_API(::glCompileShader(shader));

		return shader;
	}

	GLuint createProgram(GLuint vs, GLuint fs)
	{
		auto program = ::glCreateProgram();
		AT_ASSERT(program != 0);

		CALL_GL_API(::glAttachShader(program, vs));
		CALL_GL_API(::glAttachShader(program, fs));

		CALL_GL_API(::glLinkProgram(program));

		GLint isLinked = 0;
		CALL_GL_API(::glGetProgramiv(program, GL_LINK_STATUS, &isLinked));
		//AT_ASSERT(isLinked != 0);

		if (isLinked == 0) {
			GLint infoLen = 0;

			CALL_GL_API(::glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen));

			if (infoLen > 1) {
				char* log = (char*)malloc(infoLen);
				memset(log, 0, infoLen);

				CALL_GL_API(::glGetProgramInfoLog(program, infoLen, NULL, log));
				AT_ASSERT(false);

				free(log);
			}
		}

		return program;
	}

	bool shader::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		auto vs = createShader(pathVS, GL_VERTEX_SHADER);
		AT_VRETURN(vs != 0, false);

		auto fs = createShader(pathFS, GL_FRAGMENT_SHADER);
		AT_VRETURN(fs != 0, false);

		m_program = createProgram(vs, fs);
		AT_VRETURN(m_program != 0, false);

		m_width = width;
		m_height = height;

		return true;
	}

	void shader::prepareRender(
		const void* pixels,
		bool revert)
	{
		CALL_GL_API(::glUseProgram(m_program));
	}

	GLint shader::getHandle(const char* name)
	{
		auto handle = ::glGetUniformLocation(m_program, name);
		return handle;
	}

	void SimpleRender::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		GLfloat invScreen[4] = { 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f };
		auto hInvScreen = getHandle("invScreen");
		if (hInvScreen >= 0) {
			CALL_GL_API(::glUniform4fv(hInvScreen, 1, invScreen));
		}

		auto hRevert = getHandle("revert");
		if (hRevert >= 0) {
			CALL_GL_API(::glUniform1i(hRevert, revert ? 1 : 0));
		}

		auto hImage = getHandle("image");
		if (hImage >= 0) {
			CALL_GL_API(glUniform1i(hImage, 0));
		}
	}
}