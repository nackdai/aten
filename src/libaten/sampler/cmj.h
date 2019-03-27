#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
    // Correllated multi jittered.
    // http://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
    class CMJ AT_INHERIT(sampler) {
    public:
        AT_DEVICE_API CMJ() {}
        AT_VIRTUAL(AT_DEVICE_API ~CMJ() {});

    public:
        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const void* data = nullptr))
        {
            AT_ASSERT(false);
            init(seed, 0, 0, reinterpret_cast<const unsigned int*>(data));
        }

        AT_DEVICE_API void init(
            uint32_t index,
            uint32_t dimension,
            uint32_t scramble,
            const unsigned int* data = nullptr)
        {
            m_idx = index;
            m_dimension = dimension;
            m_scramble = scramble;
        }

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
        {
            vec2 r = sample2D();
            m_dimension++;
            return r.x;
        }

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API vec2 nextSample2D())
        {
            vec2 r = sample2D();
            m_dimension++;
            return std::move(r);
        }

        enum {
            CMJ_DIM = 16,
        };

    private:
        AT_DEVICE_API uint32_t permute(uint32_t i, uint32_t l, uint32_t p)
        {
            uint32_t w = l - 1;
            w |= w >> 1;
            w |= w >> 2;
            w |= w >> 4;
            w |= w >> 8;
            w |= w >> 16;

            do
            {
                i ^= p;
                i *= 0xe170893d;
                i ^= p >> 16;
                i ^= (i & w) >> 4;
                i ^= p >> 8;
                i *= 0x0929eb3f;
                i ^= p >> 23;
                i ^= (i & w) >> 1;
                i *= 1 | p >> 27;
                i *= 0x6935fa69;
                i ^= (i & w) >> 11;
                i *= 0x74dcb303;
                i ^= (i & w) >> 2;
                i *= 0x9e501cc3;
                i ^= (i & w) >> 2;
                i *= 0xc860a3df;
                i &= w;
                i ^= i >> 5;
            } while (i >= l);

            return (i + p) % l;
        }

        AT_DEVICE_API float randfloat(uint32_t i, uint32_t p)
        {
            i ^= p;
            i ^= i >> 17;
            i ^= i >> 10;
            i *= 0xb36534e5;
            i ^= i >> 12;
            i ^= i >> 21;
            i *= 0x93fc4795;
            i ^= 0xdf6e307f;
            i ^= i >> 17;
            i *= 1 | p >> 18;

            return i * (1.0f / 4294967808.0f);
        }

        AT_DEVICE_API vec2 cmj(int s, int n, int p)
        {
            int sx = permute(s % n, n, p * 0xa511e9b3);
            int sy = permute(s / n, n, p * 0x63d83595);
            float jx = randfloat(s, p * 0xa399d265);
            float jy = randfloat(s, p * 0x711ad6a5);

            return std::move(vec2(
                (s % n + (sy + jx) / n) / n,
                (s / n + (sx + jy) / n) / n));
        }

        AT_DEVICE_API vec2 sample2D()
        {
            int idx = permute(m_idx, CMJ_DIM * CMJ_DIM, 0xa399d265 * m_dimension * m_scramble);
            auto ret = cmj(idx, CMJ_DIM, m_dimension * m_scramble);
            return std::move(ret);
        }

    private:
        uint32_t m_idx;
        uint32_t m_dimension;
        uint32_t m_scramble;
    };
}
