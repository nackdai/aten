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

    std::shared_ptr<texture> ImageLoader::load(
        const std::string& path,
        context& ctxt,
        ImgFormat fmt/*= ImgFormat::Fmt8Bit*/)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        return load(filename, path, ctxt, fmt);
    }

    template <class TYPE>
    static void read(
        TYPE* src,
        texture* tex,
        int32_t width,
        int32_t height,
        int32_t channel,
        real norm)
    {
        int32_t skipChannel = channel;

#pragma omp parallel for
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                int32_t idx = y * width + x;
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

    std::shared_ptr<texture> ImageLoader::load(
        const std::string& tag,
        const std::string& path,
        context& ctxt,
        ImgFormat fmt/*= ImgFormat::Fmt8Bit*/)
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

        auto stored_tex = AssetManager::getTex(tag);

        if (stored_tex) {
            AT_PRINTF("There is same tag texture. [%s]\n", tag.c_str());
            return stored_tex;
        }

        std::shared_ptr<texture> tex;

        real* dst = nullptr;
        int32_t width = 0;
        int32_t height = 0;
        int32_t channels = 0;

        if (stbi_is_hdr(fullpath.c_str())) {
            auto src = stbi_loadf(fullpath.c_str(), &width, &height, &channels, 0);
            if (src) {
                tex = ctxt.CreateTexture(width, height, channels, texname.c_str());
                real norm = real(1);
                read<float>(src, tex.get(), width, height, channels, norm);

                STBI_FREE(src);
            }
        }
        else {
            void* src = nullptr;

            if (fmt == ImgFormat::Fmt8Bit) {
                src = stbi_load(fullpath.c_str(), &width, &height, &channels, 0);
            }
            else {
                src = stbi_load_16(fullpath.c_str(), &width, &height, &channels, 0);
            }

            if (src) {
                tex = ctxt.CreateTexture(width, height, channels, texname.c_str());

                if (fmt == ImgFormat::Fmt8Bit) {
                    real norm = real(1) / real(255);
                    read<stbi_uc>((stbi_uc*)src, tex.get(), width, height, channels, norm);
                }
                else {
                    real norm = real(1) / real(65535);
                    read<uint16_t>((uint16_t*)src, tex.get(), width, height, channels, norm);
                }

                STBI_FREE(src);
            }
        }

        if (tex) {
            // Store as shared_ptr
            AssetManager::registerTex(tag, tex);
        }
        else {
            AT_ASSERT(false);
            AT_PRINTF("Failed load texture. [%s]\n", fullpath.c_str());
            return nullptr;
        }

        // Get as shared_ptr
        auto ret = AssetManager::getTex(tag);
        AT_ASSERT(ret);

        return ret;
    }
}
