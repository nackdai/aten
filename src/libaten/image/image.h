#pragma once

#include <memory>
#include <string>

#include "image/texture.h"
#include "scene/host_scene_context.h"

namespace aten {
    class Image {
    private:
        Image() = delete;
        ~Image() = delete;

    public:
        static std::shared_ptr<texture> Load(
            const std::string& tag,
            const std::string& path,
            context& ctxt);
    };
}
