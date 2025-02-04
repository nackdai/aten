#pragma once

#include <string>

#include "types.h"
#include "math/vec3.h"

#define AT_DEFINE_SHADER_PARAMETER(type, param, default_value) \
    type param##_{ default_value }; \
    static constexpr const char* name_##param = #param

namespace aten {
    class shader {
    public:
        shader() = default;
        virtual ~shader() = default;

    public:
        bool init(
            int32_t width, int32_t height,
            std::string_view pathVS,
            std::string_view pathFS);

        bool init(
            int32_t width, int32_t height,
            std::string_view pathVS,
            std::string_view pathGS,
            std::string_view pathFS);

        virtual void prepareRender(
            const void* pixels,
            bool revert);

        int32_t getHandle(std::string_view name);

        uint32_t getProgramHandle() const
        {
            return m_program;
        }

        bool IsInitialized() const
        {
            return (m_program > 0);
        }

        void setUniformFloat(std::string_view name, float f);
        void setUniformInt(std::string_view name, int32_t i);
        void setUniformBool(std::string_view name, bool b)
        {
            setUniformInt(name, (int32_t)b);
        }
        void setUniformVec3(std::string_view name, const vec3& v);

    protected:
        uint32_t m_program{ 0 };
        uint32_t width_{ 0 };
        uint32_t height_{ 0 };
    };
}
