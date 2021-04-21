#pragma once

#include "aten.h"

namespace aten
{
    class SceneLoader {
    private:
        SceneLoader() {}
        ~SceneLoader() {}

    public:
        static void setBasePath(const std::string& base);

        struct ProcInfo {
            std::string type;
            Values val;
        };

        struct SceneInfo {
            std::shared_ptr<aten::scene> scene;

            std::shared_ptr<aten::camera> camera;

            std::string rendererType;
            aten::Destination dst;

            std::vector<ProcInfo> preprocs;
            std::vector<ProcInfo> postprocs;
        };

        static SceneInfo load(
            const std::string& path,
            context& ctxt);

    private:
        static void readMaterials(
            const void* xmlRoot,
            context& ctxt);
    };
}
