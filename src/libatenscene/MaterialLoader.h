#pragma once

#include <memory>

#include "aten.h"

namespace aten {
    class MaterialLoader {
        friend class SceneLoader;

    private:
        MaterialLoader() = delete;
        ~MaterialLoader() = delete;

    public:
        using MaterialCreator = std::function<material*(Values&)>;

        static void setBasePath(const std::string& base);

        static bool load(
            std::string_view path,
            context& ctxt);

        static std::shared_ptr<material> create(
            std::string_view name,
            const std::string& type,
            context& ctxt,
            Values& values);

    private:
        static void onLoad(
            const void* xmlRoot,
            context& ctxt);
    };
}
