#pragma once

#include "aten.h"
#include "AssetManager.h"

namespace aten
{
    class AssimpImporter {
    public:
        using FuncCreateMaterial = std::function<
            std::shared_ptr<aten::material> (
                const std::string& name,
                context& ctxt,
                const MaterialParameter& mtrl_param,
                const std::string& albedo,
                const std::string& nml)>;

        static bool load(
            const std::string& path,
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial func_create_mtrl);

    private:
        AssimpImporter() = default;
        ~AssimpImporter() = default;

        AssimpImporter(const AssimpImporter&) = delete;
        AssimpImporter(AssimpImporter&&) = delete;
        AssimpImporter& operator=(const AssimpImporter&) = delete;
        AssimpImporter& operator=(AssimpImporter&&) = delete;

        bool loadModel(
            const std::string& path,
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            context& ctxt,
            aten::AssetManager& asset_manager,
            FuncCreateMaterial func_create_mtrl);

        std::vector<std::string> mtrl_list_;
    };
}
