#pragma once

#include "sampler/sobol.h"
#include "sampler/random.h"

namespace aten {
	class Sobol : public random {
	public:
		Sobol(uint32_t idx) {
			m_idx = (idx == 0 ? 1 : idx);
		}
		virtual ~Sobol() {}

		virtual real next01() override final
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
		uint32_t m_idx{ 1 };
		uint32_t m_dimension{ 0 };
	};
}