#include <vector>
#include "visualizer.h"

#define CALL_GL_API(func)\
    func; \
    {\
        GLenum __gl_err__ = ::glGetError();\
        if (__gl_err__ != GL_NO_ERROR) { aten::OutputDebugString("GL Error[%x]\n", __gl_err__); AT_ASSERT(false); }\
    }

namespace aten {
	visualizer visualizer::s_instance;

	bool visualizer::init(
		int width, int height,
		const char* pathVS,
		const char* pathPS)
	{
		GLenum result = glewInit();
		AT_ASSERT(result == GLEW_OK);

		auto version = ::glGetString(GL_VERSION);
		AT_PRINTF("GL Version(%s)\n", version);

		CALL_GL_API(::glClipControl(
			GL_LOWER_LEFT,
			GL_ZERO_TO_ONE));

		CALL_GL_API(::glFrontFace(GL_CCW));

		CALL_GL_API(::glViewport(0, 0, width, height));
		CALL_GL_API(::glDepthRangef(0.0f, 1.0f));

		m_tex = createTexture(width, height);
		AT_VRETURN(m_tex != 0, false);

		auto vs = createShader(pathVS, GL_VERTEX_SHADER);
		AT_VRETURN(vs != 0, false);

		auto fs = createShader(pathVS, GL_FRAGMENT_SHADER);
		AT_VRETURN(fs != 0, false);

		m_program = createProgram(vs, fs);
		AT_VRETURN(m_program != 0, false);

		m_width = width;
		m_height = height;

		return true;
	}

	GLuint visualizer::createTexture(int width, int height)
	{
		GLuint tex = 0;

		CALL_GL_API(::glGenTextures(1, &tex));
		AT_ASSERT(tex != 0);

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, tex));

		CALL_GL_API(::glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGBA,
			width, height,
			0,
			GL_RGBA,
			GL_UNSIGNED_BYTE,
			NULL));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));

		return tex;
	}

	GLuint visualizer::createShader(const char* path, GLenum type)
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

	GLuint visualizer::createProgram(GLuint vs, GLuint fs)
	{
		auto program = ::glCreateProgram();
		AT_ASSERT(program != 0);

		CALL_GL_API(::glAttachShader(program, vs));
		CALL_GL_API(::glAttachShader(program, fs));

		CALL_GL_API(::glLinkProgram(m_program));

		GLint isLinked = 0;
		CALL_GL_API(::glGetProgramiv(m_program, GL_LINK_STATUS, &isLinked));
		AT_ASSERT(isLinked != 0);

		return program;
	}

	static inline GLint getHandle(GLuint program, const char* name)
	{
		auto handle = ::glGetUniformLocation(program, name);
		return handle;
	}

	void visualizer::beginRender()
	{
		CALL_GL_API(::glClearColor(0.0f, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glUseProgram(m_program));

		GLfloat invScreen[4] = { 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f };
		auto hInvScreen = getHandle(m_program, "invScreen");
		if (hInvScreen >= 0) {
			CALL_GL_API(::glUniform4fv(hInvScreen, 1, invScreen));
		}

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, m_tex));

		auto hImage = getHandle(m_program, "image");
		if (hImage >= 0) {
			CALL_GL_API(glUniform1i(hImage, 0));
		}
	}

	void visualizer::endRender(const void* pixels)
	{
		CALL_GL_API(::glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			0, 0,
			m_width, m_height,
			GL_RGBA,
			GL_UNSIGNED_BYTE,
			pixels));

		CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
	}
}