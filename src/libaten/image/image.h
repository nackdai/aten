#pragma once

#include <memory>
#include <string>

#include "image/texture.h"
#include "misc/color.h"
#include "scene/host_scene_context.h"

namespace aten {
    class Image {
    private:
        Image() = delete;
        ~Image() = delete;

    public:
        static std::shared_ptr<texture> Load(
            const std::string_view tag,
            const std::string_view path,
            context& ctxt,
            const AT_NAME::ColorEncoder* encoder = nullptr);

        static bool IsHdr(const std::string_view path);
    };
}
