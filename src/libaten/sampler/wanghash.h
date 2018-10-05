#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
    // Wang's hash による乱数ジェネレータ.
    class WangHash AT_INHERIT(sampler) {
    public:
        AT_DEVICE_API WangHash() {}
        WangHash(const unsigned int initial_seed)
        {
            init(initial_seed);
        }

        AT_VIRTUAL(AT_DEVICE_API ~WangHash() {})

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const unsigned int* data = nullptr))
        {
            m_seed = seed;
        }

        // [0, 1]
        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
        {
            auto ret = (real)next(m_seed) / UINT_MAX;
            m_seed += 1;
            return ret;
        }

        // NOTE
        // https://gist.github.com/badboy/6267743
        static AT_DEVICE_API uint32_t next(unsigned int seed)
        {
            uint32_t key = 1664525U * seed + 1013904223U;

            key = (key ^ 61) ^ (key >> 16);
            key = key + (key << 3);
            key = key ^ (key >> 4);
            key = key * 0x27d4eb2d;
            key = key ^ (key >> 15);

            return key;
        }

    private:
        unsigned int m_seed;
    };
}
