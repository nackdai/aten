#include "hdr/hdr.h"

namespace aten
{
    struct HDRPixel {
        unsigned char r, g, b, e;

        HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
            r(r_), g(g_), b(b_), e(e_) {};

        unsigned char get(int32_t idx) {
            switch (idx) {
            case 0: return r;
            case 1: return g;
            case 2: return b;
            case 3: return e;
            } return 0;
        }
    };

    // doubleのRGB要素を.hdrフォーマット用に変換.
    static HDRPixel get_hdr_pixel(const vec4& color)
    {
        double d = std::max(color.x, std::max(color.y, color.z));

        if (d <= 1e-32) {
            return HDRPixel();
        }

        int32_t e;
        double m = frexp(d, &e); // d = m * 2^e
        d = m * 256.0 / d;

        return HDRPixel(
            static_cast<uint8_t>(color.x * d),
            static_cast<uint8_t>(color.y * d),
            static_cast<uint8_t>(color.z * d),
            e + 128);
    }

    bool HDRExporter::save(
        const std::string& filename,
        const vec4* image,
        const int32_t width, const int32_t height)
    {
        FILE *fp = fopen(filename.c_str(), "wb");
        if (fp == NULL) {
            AT_PRINTF("Error: %s¥n", filename.c_str());
            return false;
        }

        // .hdrフォーマットに従ってデータを書きだす.
        // ヘッダ.
        unsigned char ret = 0x0a;
        fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
        fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
        fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);
        fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

        // 輝度値書き出し.
        fprintf(fp, "-Y %d +X %d%c", height, width, ret);

        for (int32_t i = height - 1; i >= 0; i--) {
            std::vector<HDRPixel> line;
            for (int32_t j = 0; j < width; j++) {
                HDRPixel p = get_hdr_pixel(image[j + i * width]);
                line.push_back(p);
            }

            fprintf(fp, "%c%c", 0x02, 0x02);
            fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);

            for (int32_t i = 0; i < 4; i++) {
                for (int32_t cursor = 0; cursor < width;) {
                    const int32_t cursor_move = std::min(127, width - cursor);
                    fprintf(fp, "%c", cursor_move);

                    for (int32_t j = cursor; j < cursor + cursor_move; j++) {
                        fprintf(fp, "%c", line[j].get(i));
                    }

                    cursor += cursor_move;
                }
            }
        }

        fclose(fp);

        return true;
    }
}
