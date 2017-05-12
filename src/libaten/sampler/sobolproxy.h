#pragma once

#include "sampler/sobol.h"
#include "sampler/sampler.h"

namespace aten {
	// NOTE
	// The code of sobol is taken from: http://gruenschloss.org/sobol/kuo-2d-proj-single-precision.zip

	class Sobol : public sampler {
	public:
		Sobol(uint32_t idx)
		{
			init(idx);
		}
		virtual ~Sobol() {}

		virtual void init(uint32_t seed) override final
		{
			m_idx = (seed == 0 ? 1 : seed);
			m_dimension = 0;
		}

		virtual real nextSample() override final
		{
			if (m_dimension >= sobol::Matrices::num_dimensions) {
				AT_ASSERT(false);
				return aten::drand48();
			}

			auto ret = sobol::sample(m_idx, m_dimension);
			m_dimension += 1;

			return ret;
		}

	private:
		uint32_t m_idx;
		uint32_t m_dimension;
	};
}