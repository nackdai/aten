#pragma once

#include "aten.h"

namespace aten
{
    class AssimpImporter {
    public:
        using FuncCreateMaterial = std::function<
            std::shared_ptr<aten::material> (
                std::string_view name,
                context& ctxt,
                const MaterialParameter& mtrl_param,
                const std::string& albedo,
                const std::string& nml)>;

        static bool load(
            std::string_view path,
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            context& ctxt,
            FuncCreateMaterial func_create_mtrl);

    private:
        AssimpImporter() = default;
        ~AssimpImporter() = default;

        AssimpImporter(const AssimpImporter&) = delete;
        AssimpImporter(AssimpImporter&&) = delete;
        AssimpImporter& operator=(const AssimpImporter&) = delete;
        AssimpImporter& operator=(AssimpImporter&&) = delete;

        bool loadModel(
            std::string_view path,
            std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
            context& ctxt,
            FuncCreateMaterial func_create_mtrl);

        std::vector<std::string> mtrl_list_;
    };
}
