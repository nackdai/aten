#pragma once

#include "types.h"
#include "math/vec3.h"

namespace aten {
    class shader {
    public:
        shader() {}
        virtual ~shader() {}

    public:
        bool init(
            int32_t width, int32_t height,
            const char* pathVS,
            const char* pathFS);

        bool init(
            int32_t width, int32_t height,
            const char* pathVS,
            const char* pathGS,
            const char* pathFS);

        virtual void prepareRender(
            const void* pixels,
            bool revert);

        //GLint getHandle(const char* name);
        int32_t getHandle(const char* name);

        uint32_t getProgramHandle() const
        {
            return m_program;
        }

        bool isValid() const
        {
            return (m_program > 0);
        }

        void setUniformFloat(const char* name, real f);
        void setUniformInt(const char* name, int32_t i);
        void setUniformBool(const char* name, bool b)
        {
            setUniformInt(name, (int32_t)b);
        }
        void setUniformVec3(const char* name, const vec3& v);

    protected:
        //GLuint m_program{ 0 };
        uint32_t m_program{ 0 };
        uint32_t m_width{ 0 };
        uint32_t m_height{ 0 };
    };
}
