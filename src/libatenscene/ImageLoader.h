#pragma once

#include <memory>
#include <string>

#include "image/image.h"
#include "scene/host_scene_context.h"

namespace aten {
    class ImageLoader {
    private:
        ImageLoader() = delete;
        ~ImageLoader() = delete;

        static std::string base_path;

    public:
        static void setBasePath(const std::string& base);

        static std::shared_ptr<texture> load(
            const std::string& path,
            context& ctxt);

        static std::shared_ptr<texture> load(
            const std::string& tag,
            const std::string& path,
            context& ctxt);
    };
}
