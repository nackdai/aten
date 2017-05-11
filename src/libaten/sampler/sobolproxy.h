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
			m_param.idx = (seed == 0 ? 1 : seed);
			m_param.dimension = 0;
		}

		virtual real nextSample() override final
		{
			if (m_param.dimension >= sobol::Matrices::num_dimensions) {
				AT_ASSERT(false);
				return aten::drand48();
			}

			auto ret = sobol::sample(m_param.idx, m_param.dimension);
			m_param.dimension += 1;

			return ret;
		}

	private:
		SamplerParameter m_param;
	};
}