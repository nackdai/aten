#pragma once

#include <string>

#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
    class IParamEditor {
    public:
        IParamEditor() = default;
        virtual ~IParamEditor() = default;

        IParamEditor(const IParamEditor&) = delete;
        IParamEditor(IParamEditor&&) = delete;
        IParamEditor& operator=(const IParamEditor&) = delete;
        IParamEditor& operator=(IParamEditor&&) = delete;

    public:
        virtual bool edit(std::string_view name, float& param, float _min = 0.0F, float _max = 1.0F) = 0;
        virtual bool edit(std::string_view name, bool& param) = 0;
        virtual bool edit(std::string_view name, vec3& param) = 0;
        virtual bool edit(std::string_view name, vec4& param) = 0;
        virtual bool edit(std::string_view name, const char* const* elements, size_t size, int32_t& param) = 0;

        virtual bool CollapsingHeader(std::string_view name)
        {
            return true;
        }

        void editTex(std::string_view name, int32_t texid)
        {
        }

    protected:
        virtual void edit(std::string_view name, std::string_view str) = 0;
    };
}

#if defined(_WIN32) || defined(_WIN64)
#define AT_EDIT_MATERIAL_PARAM(e, param, name)    (e)->edit(#name, param.##name)
#define AT_EDIT_MATERIAL_PARAM_RANGE(e, param, name, _min, _max)    (e)->edit(#name, param.##name, _min, _max)
#define AT_EDIT_MATERIAL_PARAM_TEXTURE(e, param, name)    (e)->editTex(#name, param.##name)
#else
// TODO
// For linux, to avoid token concat error.
#define AT_EDIT_MATERIAL_PARAM(e, param, name)  false
#define AT_EDIT_MATERIAL_PARAM_RANGE(e, param, name, _min, _max)    false
#define AT_EDIT_MATERIAL_PARAM_TEXTURE(e, param, name)
#endif
