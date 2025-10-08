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
            std::string_view path_vs,
            std::string_view path_fs);

        bool init(
            int32_t width, int32_t height,
            std::string_view path_vs,
            std::string_view path_gs,
            std::string_view path_fs);

        virtual void PrepareRender(
            const void* pixels,
            bool revert);

        int32_t GetHandle(std::string_view name);

        uint32_t GetProgramHandle() const
        {
            return program_;
        }

        bool IsInitialized() const
        {
            return (program_ > 0);
        }

        void SetUniformFloat(std::string_view name, float f);
        void SetUniformInt(std::string_view name, int32_t i);
        void SetUniformBool(std::string_view name, bool b)
        {
            SetUniformInt(name, static_cast<int32_t>(b));
        }
        void SetUniformVec3(std::string_view name, const vec3& v);

    protected:
        uint32_t program_{ 0 };
        uint32_t width_{ 0 };
        uint32_t height_{ 0 };
    };
}
