#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
    // Xor-Shift による乱数ジェネレータ.
    class XorShift : public sampler {
    public:
        XorShift() {}
        XorShift(const uint32_t initial_seed)
        {
            init(initial_seed);
        }

        virtual ~XorShift() {}

        uint32_t next()
        {
            const uint32_t t = m_seed_[0] ^ (m_seed_[0] << 11);
            m_seed_[0] = m_seed_[1];
            m_seed_[1] = m_seed_[2];
            m_seed_[2] = m_seed_[3];
            return m_seed_[3] = (m_seed_[3] ^ (m_seed_[3] >> 19)) ^ (t ^ (t >> 8));
        }

        // [0, 1]
        virtual real nextSample() override final
        {
            return static_cast<real>(next() / UINT_MAX);
        }

        virtual void init(uint32_t initial_seed, const void* data = nullptr) override final
        {
            uint32_t s = initial_seed;
            for (int32_t i = 1; i <= 4; i++) {
                m_seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
            }
        }

    private:
        uint32_t m_seed_[4];
    };
}
