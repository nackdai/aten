#include <cmdline.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "aten.h"

struct Options {
};

bool parseOption(
    int argc, char* argv[],
    Options& opt)
{
    cmdline::parser cmd;

    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false);
        cmd.add<std::string>("type", 't', "export type(m = model, a = animation)", true, "m");
        cmd.add<std::string>("base", 'b', "input filename for animation base model", false);
        cmd.add("gpu", 'g', "export for gpu skinning");

        cmd.add("help", '?', "print usage");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error_full() << std::endl << cmd.usage();
        return false;
    }

    return true;
}

uint32_t genRandomUint(
    uint32_t input,
    uint32_t scramble = 0)
{
    input ^= scramble;
    input ^= input >> 17;
    input ^= input >> 10;
    input *= 0xb36534e5;
    input ^= input >> 12;
    input ^= input >> 21;
    input *= 0x93fc4795;
    input ^= 0xdf6e307f;
    input ^= input >> 17;
    input *= 1 | scramble >> 18;

    return input;
}

float getRandomFloat(
    uint32_t input,
    uint32_t scramble = 0)
{
    auto rui = genRandomUint(input, scramble);
    rui = (0x7F << 23) | (rui >> 9);

    float ret = *(float*)&rui;
    ret -= 1.0f;

    return ret;
}

uint32_t computeHash(uint32_t input)
{
    input = ~input + (input << 15);
    input = input ^ (input >> 12);
    input = input + (input << 2);
    input = input ^ (input >> 4);
    input = input * 2057;
    input = input ^ (input >> 16);

    return input;
}

class Buffer{
public:
    Buffer(int w, int h) : m_w(w), m_h(h)
    {
        m_buffer.resize(m_w * m_h);
    }
    ~Buffer() {}

public:
    aten::vec3& at(int x, int y)
    {
        AT_ASSERT(x < m_w);
        AT_ASSERT(y < m_h);

        int pos = y * m_w + x;
        return m_buffer[pos];
    }

    std::vector<aten::vec3>& getBuffer()
    {
        return m_buffer;
    }

    int width() const
    {
        return m_w;
    }

    int height() const
    {
        return m_h;
    }

private:
    std::vector<aten::vec3> m_buffer;
    int m_w{ 0 };
    int m_h{ 0 };
};

float compute(
    Buffer& buffer,
    int ix, int iy,
    int width, int height,
    float sigma_i, float sigma_s)
{
    std::vector<float> sum(height);

#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
    for (int y = -7; y <= 7; ++y)
    {
        for (int x = -7; x <= 7; ++x)
        {
            int sx = ix + x;
            if (sx < 0) {
                sx += width;
            }
            if (sx >= width) {
                sx -= width;
            }

            int sy = iy + y;
            if (sy < 0) {
                sy += height;
            }
            if (sy >= height) {
                sy -= height;
            }

            int dx = abs(ix - sx);
            if (dx > width / 2) {
                dx = width - dx;
            }

            int dy = abs(iy - sy);
            if (dy > height / 2) {
                dy = height - dy;
            }

            const float a = (dx * dx + dy * dy) / (sigma_i * sigma_i);

#if 0
            // One dimension.
            const float b = sqrt(abs(buffer.at(ix, iy).x - buffer.at(sx, sy).x)) / (sigma_s * sigma_s);
#else
            const float da = abs(buffer.at(ix, iy)[0] - buffer.at(sx, sy)[0]);
            const float db = abs(buffer.at(ix, iy)[1] - buffer.at(sx, sy)[1]);

            const float b = sqrtf(da * da + db * db) / (sigma_s * sigma_s);
#endif

            sum[sy] += exp(-a - b);
        }
    }

    float total = 0;
    for (int y = 0; y < height; y++) {
        total += sum[y];
    }

    return total;
}

float compute(
    Buffer& buffer,
    int width, int height,
    float sigma_i, float sigma_s)
{
    float ret = 0.0f;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            ret += compute(buffer, x, y, width, height, sigma_i, sigma_s);
        }
    }

    return ret;
}

int main(int argc, char* argv[])
{
    aten::timer::init();
    aten::timer time;

    int W = 256;
    int H = 256;

    const float sigma_i = 2.1f;
    const float sigma_s = 1.0f;

    Buffer blueNoise(W, H);
    Buffer proposalBuffer(W, H);

    auto seedHash_0 = computeHash(0);
    auto seedHash_1 = computeHash(1);

    time.begin();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int i = y * W + x;

            float r0 = getRandomFloat(i * 2 + 0, seedHash_0 ^ seedHash_1);
            float r1 = getRandomFloat(i * 2 + 1, seedHash_0 ^ seedHash_1);

            blueNoise.at(x, y).x = r0;
            blueNoise.at(x, y).y = r1;
            blueNoise.at(x, y).z = 0.0f;

            proposalBuffer.at(x, y).x = r0;
            proposalBuffer.at(x, y).y = r1;
            proposalBuffer.at(x, y).z = 0.0f;
        }
    }

    float initialDistribution = compute(blueNoise, W, H, sigma_i, sigma_s);

    float blueNoiseDistribution = initialDistribution;

    auto elapsed = time.end();
    AT_PRINTF("Initial Exec %f[ms]\n", elapsed);

    const int iteration = 200;

    // Ä‚«‚È‚Ü‚µ–@.

    static const char* MS = "ms";
    static const char* SEC = "sec";
    static const int LOG_TIMING = 3;

    time.begin();

    for (int i = 0; i < iteration; i++) {
        if (i > 0 && i % LOG_TIMING == 0) {
            auto elapsedFromPrev = time.end();
            elapsed += elapsedFromPrev;
            float ratio = real(100) * real(i) / iteration;

            auto allElapsed = elapsed;

            const char* TimeElapsed = (elapsedFromPrev > real(1000) ? SEC : MS);
            const char* TimeAllElapsed = (allElapsed > real(1000) ? SEC : MS);

            elapsedFromPrev = (elapsedFromPrev > real(1000) ? elapsedFromPrev / real(1000) : elapsedFromPrev);
            allElapsed = (allElapsed > real(1000) ? allElapsed / real(1000) : allElapsed);

            AT_PRINTF(
                "Percent:%f[%%] Elapsed:%f(%s) AllElapsed:%f(%s)\n",
                ratio, elapsedFromPrev, TimeElapsed, allElapsed, TimeAllElapsed);

            time.begin();
        }

        int y0 = static_cast<int>(getRandomFloat(i * 4 + 0) * H);
        int y1 = static_cast<int>(getRandomFloat(i * 4 + 1) * H);
        int x0 = static_cast<int>(getRandomFloat(i * 4 + 2) * W);
        int x1 = static_cast<int>(getRandomFloat(i * 4 + 3) * W);

        std::swap(proposalBuffer.at(x0, y0), proposalBuffer.at(x1, y1));

        float proposalDistribution = compute(proposalBuffer, W, H, sigma_i, sigma_s);

        if (proposalDistribution > blueNoiseDistribution) {
            proposalBuffer.at(x0, y0) = blueNoise.at(x0, y0);
            proposalBuffer.at(x1, y1) = blueNoise.at(x1, y1);
            continue;
        }

        blueNoiseDistribution = proposalDistribution;

        blueNoise.at(x0, y0) = proposalBuffer.at(x0, y0);
        blueNoise.at(x1, y1) = proposalBuffer.at(x1, y1);
    }

#if 0
    // Check length.
    for (int y = 0; y < H; y++) {
        for (int x = 1; x < W - 1; x++) {
            const auto& v0 = blueNoise.at(x, y);
            const auto& v1 = blueNoise.at(x - 1, y);

            auto l = aten::length(v0 - v1);
            AT_PRINTF("%f\n", l);
        }
    }
#endif

    // Export to image data.

    return 1;
}
