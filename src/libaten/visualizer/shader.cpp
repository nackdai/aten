#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "visualizer/atengl.h"
#include "visualizer/shader.h"

namespace aten {
    GLuint createShader(std::string_view path, GLenum type)
    {
        std::filesystem::path p = path;

        if (!std::filesystem::exists(p)) {
            AT_ASSERT(false);
            AT_PRINTF("%s doesn't exist.", path.data());
            return 0;
        }

        const auto size = std::filesystem::file_size(p);

        std::ifstream ifs(path.data(), std::ios_base::in);
        if (!ifs) {
            AT_ASSERT(false);
            AT_PRINTF("Can't open %s.", path.data());
            return 0;
        }

        std::vector<char> program(size + 1);
        ifs.read(program.data(), size);

        ifs.close();

        CALL_GL_API(auto shader = ::glCreateShader(type));
        AT_ASSERT(shader != 0);

        const auto program_ptr = program.data();
        const auto pp = &program_ptr;

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
        std::string_view pathVS,
        std::string_view pathFS)
    {
        auto vs = createShader(pathVS, GL_VERTEX_SHADER);
        AT_VRETURN(vs != 0, false);

        auto fs = createShader(pathFS, GL_FRAGMENT_SHADER);
        AT_VRETURN(fs != 0, false);

        m_program = createProgram(vs, fs);
        AT_VRETURN(m_program != 0, false);

        width_ = width;
        height_ = height;

        return true;
    }

    bool shader::init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathGS,
        std::string_view pathFS)
    {
        auto vs = createShader(pathVS, GL_VERTEX_SHADER);
        AT_VRETURN(vs != 0, false);

        auto fs = createShader(pathFS, GL_FRAGMENT_SHADER);
        AT_VRETURN(fs != 0, false);

        auto gs = createShader(pathGS, GL_GEOMETRY_SHADER);
        AT_VRETURN(gs != 0, false);

        m_program = createProgram(vs, fs, gs);
        AT_VRETURN(m_program != 0, false);

        width_ = width;
        height_ = height;

        return true;
    }

    void shader::prepareRender(
        const void* pixels,
        bool revert)
    {
        CALL_GL_API(::glUseProgram(m_program));
    }

    int32_t shader::getHandle(std::string_view name)
    {
        auto handle = ::glGetUniformLocation(m_program, name.data());
        return handle;
    }

    void shader::setUniformFloat(std::string_view name, float f)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform1f(handle, (float)f));
        }
    }

    void shader::setUniformInt(std::string_view name, int32_t i)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform1i(handle, i));
        }
    }

    void shader::setUniformVec3(std::string_view name, const vec3& v)
    {
        auto handle = getHandle(name);
        if (handle >= 0) {
            CALL_GL_API(::glUniform3f(handle, v.x, v.y, v.z));
        }
    }
}
