#pragma once

#include "AssetManager.h"

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

        static std::shared_ptr<aten::PolygonObject> load(
            std::string_view path,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool needComputeNormalOntime = false);
        static std::shared_ptr<aten::PolygonObject> load(
            std::string_view tag,
            std::string_view path,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool needComputeNormalOntime = false);

        static void load(
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            std::string_view path,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
        static void load(
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            std::string_view tag,
            std::string_view path,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial callback_crate_mtrl = nullptr,
            bool willSeparate = false,
            bool needComputeNormalOntime = false);
    };
}
