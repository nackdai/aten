#pragma once

#include <memory>
#include <string>

#include "texture/texture.h"
#include "scene/context.h"

namespace aten {
    class ImageLoader {
    private:
        ImageLoader() = delete;
        ~ImageLoader() = delete;

    public:
        enum ImgFormat {
            Fmt8Bit,
            Fmt16Bit,
        };

        static void setBasePath(const std::string& base);

        static std::shared_ptr<texture> load(
            const std::string& path,
            context& ctxt,
            ImgFormat fmt = ImgFormat::Fmt8Bit);

        static std::shared_ptr<texture> load(
            const std::string& tag,
            const std::string& path,
            context& ctxt,
            ImgFormat fmt = ImgFormat::Fmt8Bit);
    };
}
