#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.h"

namespace aten {
    namespace {
        template <class TYPE>
        static void ReadPixleWithConversion(
            const TYPE* src,
            texture* tex,
            const int32_t width,
            const int32_t height,
            const int32_t channel,
            const float norm)
        {
#pragma omp parallel for
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    int32_t idx = y * width + x;
                    idx *= channel;

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
    }

    std::shared_ptr<texture> Image::Load(
        const std::string_view tag,
        const std::string_view path,
        context& ctxt)
    {
        auto stored_tex = ctxt.GetTextureByName(tag);
        if (stored_tex) {
            AT_PRINTF("There is same tag texture. [%s]\n", tag);
            return stored_tex;
        }

        std::shared_ptr<texture> tex;

        float* dst = nullptr;
        int32_t width = 0;
        int32_t height = 0;
        int32_t channels = 0;

        if (stbi_is_hdr(path.data())) {
            auto src = stbi_loadf(path.data(), &width, &height, &channels, 0);
            if (src) {
                tex = ctxt.CreateTexture(width, height, channels, tag);

                constexpr auto norm = 1.0F;
                ReadPixleWithConversion(src, tex.get(), width, height, channels, norm);

                STBI_FREE(src);
            }
        }
        else {
            auto src = stbi_load(path.data(), &width, &height, &channels, 0);
            if (src) {
                tex = ctxt.CreateTexture(width, height, channels, tag);

                constexpr auto norm = 1.0F / 255;
                ReadPixleWithConversion(src, tex.get(), width, height, channels, norm);

                STBI_FREE(src);
            }
        }

        if (!tex) {
            AT_ASSERT(false);
            AT_PRINTF("Failed load texture. [%s]\n", path.data());
            return nullptr;
        }

        return tex;
    }
}
