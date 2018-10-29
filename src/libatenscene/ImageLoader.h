#pragma once

#include <string>
#include "texture/texture.h"
#include "scene/context.h"

namespace aten {
    class ImageLoader {
    private:
        ImageLoader() {}
        ~ImageLoader() {}

    public:
        static void setBasePath(const std::string& base);

        static texture* load(
            const std::string& path,
            context& ctxt);

        static texture* load(
            const std::string& tag, 
            const std::string& path,
            context& ctxt);
    };
}