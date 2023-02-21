#include <string.h>
#include <vector>

#include "visualizer/atengl.h"
#include "visualizer/shader.h"

namespace aten {
    GLuint createShader(const char* path, GLenum type)
    {
        FILE* fp = fopen(path, "rb");
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

    GLuint createProgram(GLuint vs, GLuint fs, GLuint gs = 0)
    {
        auto program = ::glCreateProgram();
        AT_ASSERT(program != 0);

        CALL_GL_API(::glAttachShader(program, vs));
        CALL_GL_API(::glAttachShader(program, fs));

        // For geometry shader.
        if (gs > 0) {
            CALL_GL_API(::glAttachShader(program, gs));
        }

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

                AT_PRINTF("%s\n", log);

                free(log);
            }
        }

        return program;
    }

    bool shader::init(
        int32_t width, int32_t height,
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

    bool shader::init(
        int32_t width, int32_t height,
        const char* pathVS,
        const char* pathGS,
        const char* pathFS)
    {
        auto vs = createShader(pathVS, GL_VERTEX_SHADER);
        AT_VRETURN(vs != 0, false);

        auto fs = createShader(pathFS, GL_FRAGMENT_SHADER);
        AT_VRETURN(fs != 0, false);

        auto gs = createShader(pathGS, GL_GEOMETRY_SHADER);
        AT_VRETURN(gs != 0, false);

        m_program = createProgram(vs, fs, gs);
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

    void shader::setUniformFloat(const char* name, real f)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform1f(handle, (float)f));
        }
    }

    void shader::setUniformInt(const char* name, int32_t i)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform1i(handle, i));
        }
    }

    void shader::setUniformVec3(const char* name, const vec3& v)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform3f(handle, v.x, v.y, v.z));
        }
    }
}
