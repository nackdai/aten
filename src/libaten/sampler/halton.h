#pragma once

#include <vector>
#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
    class Halton : public sampler {
    private:
        static std::vector<uint32_t> PrimeNumbers;

    public:
        static const uint32_t MaxPrimeNumbers = 10000000;

        // ëfêîê∂ê¨.
        static void makePrimeNumbers(uint32_t maxNumber = MaxPrimeNumbers);

    public:
        Halton(uint32_t idx)
        {
            init(idx);
        }
        virtual ~Halton() {}

        virtual void init(uint32_t seed, const void* data = nullptr) override final
        {
            m_idx = (seed == 0 ? 1 : seed);
            m_dimension = 0;
        }

        // [0, 1]
        virtual real nextSample() override final;

    private:
        uint32_t m_idx;
        uint32_t m_dimension;
    };
}
