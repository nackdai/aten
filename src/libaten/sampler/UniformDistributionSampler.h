#pragma once

#include "sampler/xorshift.h"
#include "sampler/sampler.h"

namespace aten {
	class UniformDistributionSampler : public sampler {
	public:
		UniformDistributionSampler(random* rnd)
			: m_rnd(rnd)
		{}
		~UniformDistributionSampler() {}

		real nextSample()
		{
			auto ret = m_rnd->next01();
			return ret;
		}

	private:
		random* m_rnd;
	};
}
