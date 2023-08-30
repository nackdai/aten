#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
    // Wang's hash による乱数ジェネレータ.
    class WangHash AT_INHERIT(sampler_interface) {
    public:
        WangHash() = default;
        WangHash(const uint32_t initial_seed)
        {
            init(initial_seed);
        }

        AT_VIRTUAL(~WangHash() {})

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const void* data = nullptr))
        {
            m_seed = seed;
        }

        // [0, 1]
        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
        {
            auto ret = static_cast<real>(next(m_seed) / UINT_MAX);
            m_seed += 1;
            return ret;
        }

        // NOTE
        // https://gist.github.com/badboy/6267743
        static AT_DEVICE_API uint32_t next(uint32_t seed)
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
        uint32_t m_seed{ 0 };
    };
}
