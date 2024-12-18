#include <random>
#include <algorithm>
#include "sampler/samplerinterface.h"

namespace aten {
    static std::vector<uint32_t> g_random;

    void initSampler(
        int32_t width, int32_t height,
        int32_t seed/*= 0*/)
    {
        // TODO
        ::srand(seed);

        g_random.resize(width * height);
        std::mt19937 rand_src(seed);
        std::generate(g_random.begin(), g_random.end(), rand_src);
    }

    const std::vector<uint32_t>& getRandom()
    {
        return g_random;
    }
    uint32_t getRandom(uint32_t idx)
    {
        return g_random[idx % g_random.size()];
    }
}
