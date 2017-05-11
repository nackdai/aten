#pragma once

#include "types.h"
#include "sampler/sampler.h"

namespace aten {
	// Xor-Shift による乱数ジェネレータ.
	class XorShift : public sampler {
	public:
		XorShift() {}
		XorShift(const unsigned int initial_seed)
		{
			init(initial_seed);
		}

		virtual ~XorShift() {}

		unsigned int next()
		{
			const unsigned int t = m_param.seed_[0] ^ (m_param.seed_[0] << 11);
			m_param.seed_[0] = m_param.seed_[1];
			m_param.seed_[1] = m_param.seed_[2];
			m_param.seed_[2] = m_param.seed_[3];
			return m_param.seed_[3] = (m_param.seed_[3] ^ (m_param.seed_[3] >> 19)) ^ (t ^ (t >> 8));
		}

		// [0, 1]
		virtual real nextSample() override final
		{
			//return (real)next() / (UINT_MAX + 1.0);
			return (real)next() / UINT_MAX;
		}

		virtual void init(uint32_t initial_seed) override final
		{
			unsigned int s = initial_seed;
			for (int i = 1; i <= 4; i++) {
				m_param.seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
			}
		}

	private:
		SamplerParameter m_param;
	};
}
