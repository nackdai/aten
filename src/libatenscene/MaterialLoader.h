#pragma once

#include <memory>

#include "aten.h"

namespace aten {
    class MaterialLoader {
        friend class SceneLoader;

    private:
        MaterialLoader();
        ~MaterialLoader();

    public:
        using MaterialCreator = std::function<material*(Values&)>;

        static void setBasePath(const std::string& base);

        static bool addCreator(std::string type, MaterialCreator creator);

        static bool load(
            std::string_view path,
            context& ctxt);

        static std::shared_ptr<material> create(
            const std::string& type,
            context& ctxt,
            Values& values);

    private:
#ifdef USE_JSON
        static void onLoad(const std::string& strJson);
#else
        static void onLoad(
            const void* xmlRoot,
            context& ctxt);
#endif
    };
}
