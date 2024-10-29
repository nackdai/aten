#pragma once

#include "aten.h"

namespace aten
{
    class ObjLoader {
    private:
        ObjLoader() = delete;
        ~ObjLoader() = delete;

    public:
        using FuncCreateMaterial = std::function<
            std::shared_ptr<material> (
                std::string_view name,
                context& ctxt,
                MaterialType type,
                const vec3& mtrl_clr,
                const std::string& albedo,
                const std::string& nml)>;

        static void setBasePath(std::string_view base);

        static std::shared_ptr<aten::PolygonObject> LoadFirstObj(
            std::string_view path,
            context& ctxt,
            FuncCreateMaterial callback_create_mtrl = nullptr,
            bool need_compute_normal_on_the_fly = false);

        static std::vector<std::shared_ptr<aten::PolygonObject>> Load(
            std::string_view path,
            context& ctxt,
            FuncCreateMaterial callback_create_mtrl = nullptr,
            bool will_register_shape_as_separate_obj = false,
            bool need_compute_normal_on_the_fly = false);

    private:
        static std::vector<std::shared_ptr<aten::PolygonObject>> OnLoad(
            std::string_view path,
            context& ctxt,
            FuncCreateMaterial callback_create_mtrl = nullptr,
            bool will_register_shape_as_separate_obj = false,
            bool need_compute_normal_on_the_fly = false);
    };
}
