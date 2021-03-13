#pragma once

#include "aten.h"

namespace aten
{
    class ObjLoader {
    private:
        ObjLoader() {}
        ~ObjLoader() {}

    public:
        using FuncCreateMaterial = std::function<material* (const std::string& name, context& ctxt, MaterialType type, const vec3& mtrl_clr, const std::string& albedo, const std::string& nml)>;

        static void setBasePath(const std::string& base);

        static object* load(
            const std::string& path,
            context& ctxt,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool needComputeNormalOntime = false);
        static object* load(
            const std::string& tag,
            const std::string& path,
            context& ctxt,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool needComputeNormalOntime = false);

        static void load(
            std::vector<object*>& objs,
            const std::string& path,
            context& ctxt,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
        static void load(
            std::vector<object*>& objs,
            const std::string& tag,
            const std::string& path,
            context& ctxt,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
    };
}
