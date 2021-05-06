#pragma once

#include "sampler/sobol.h"
#include "sampler/samplerinterface.h"

namespace aten {
    // NOTE
    // The code of sobol is taken from: http://gruenschloss.org/sobol/kuo-2d-proj-single-precision.zip

    class Sobol AT_INHERIT(sampler) {
    public:
        Sobol() = default;
        Sobol(uint32_t idx)
        {
            init(idx);
        }
        AT_VIRTUAL(~Sobol() = default;)

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const void* data = nullptr))
        {
            m_idx = (seed == 0 ? 1 : seed);
            m_dimension = 0;
#ifdef __CUDACC__
            m_matrices = reinterpret_cast<const unsigned int*>(data);
#else
            m_matrices = data ? reinterpret_cast<const unsigned int*>(data) : sobol::Matrices::matrices;
#endif
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
#ifdef __CUDACC__
            m_matrices = data ? data : m_matrices;
#else
            m_matrices = data ? data : sobol::Matrices::matrices;
#endif
        }

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
        {
#ifndef __CUDACC__
            if (m_dimension >= sobol::Matrices::num_dimensions) {
                AT_ASSERT(false);
                return aten::drand48();
            }
#endif

            auto ret = sobol::sample(m_matrices, m_idx, m_dimension, m_scramble);
            m_dimension += 1;

            return ret;
        }

    private:
        uint32_t m_idx{ 0 };
        uint32_t m_dimension{ 0 };
        uint32_t m_scramble{ 0 };
        const unsigned int* m_matrices{ nullptr };
    };
}
