#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "ImageLoader.h"
#include "AssetManager.h"
#include "utility.h"

namespace aten {
    static std::string g_base;

    void ImageLoader::setBasePath(const std::string& base)
    {
        g_base = removeTailPathSeparator(base);
    }

    texture* ImageLoader::load(
        const std::string& path,
        context& ctxt)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        auto tex = load(filename, path, ctxt);

        return tex;
    }

    template <typename TYPE>
    static void read(
        TYPE* src,
        texture* tex,
        int width,
        int height,
        int channel,
        real norm)
    {
        int skipChannel = channel;

#pragma omp parallel for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                idx *= skipChannel;

                switch (channel) {
                case 4:
                    (*tex)(x, y, 3) = src[idx + 3] * norm;
                case 3:
                    (*tex)(x, y, 2) = src[idx + 2] * norm;
                case 2:
                    (*tex)(x, y, 1) = src[idx + 1] * norm;
                case 1:
                    (*tex)(x, y, 0) = src[idx + 0] * norm;
                    break;
                }
            }
        }
    }

    texture* ImageLoader::load(
        const std::string& tag, 
        const std::string& path,
        context& ctxt)
    {
        std::string fullpath = path;
        if (!g_base.empty()) {
            fullpath = g_base + "/" + fullpath;
        }

        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        std::string texname = filename + extname;

        texture* tex = AssetManager::getTex(tag);

        if (tex) {
            AT_PRINTF("There is same tag texture. [%s]\n", tag.c_str());
            return tex;
        }

        real* dst = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;

        if (stbi_is_hdr(fullpath.c_str())) {
            auto src = stbi_loadf(fullpath.c_str(), &width, &height, &channels, 0);
            if (src) {
                tex = ctxt.createTexture(width, height, channels, texname.c_str());
                real norm = real(1);
                read<float>(src, tex, width, height, channels, norm);

                STBI_FREE(src);
            }
        }
        else {
            auto src = stbi_load(fullpath.c_str(), &width, &height, &channels, 0);
            if (src) {
                tex = ctxt.createTexture(width, height, channels, texname.c_str());
                real norm = real(1) / real(255);

                read<stbi_uc>(src, tex, width, height, channels, norm);

                STBI_FREE(src);
            }
        }

        if (tex) {
            AssetManager::registerTex(tag, tex);
        }
        else {
            AT_ASSERT(false);
            AT_PRINTF("Failed load texture. [%s]\n", fullpath.c_str());
            return nullptr;
        }

        return tex;
    }
}