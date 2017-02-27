#pragma once

#include "sampler/random.h"
#include "sampler/sampler.h"

namespace aten {
	class UniformDistributionSampler : public sampler {
	public:
		UniformDistributionSampler(random* rnd)
			: m_rnd(rnd)
		{}
		~UniformDistributionSampler() {}

		virtual real nextSample() override final
		{
			auto ret = m_rnd->next01();
			return ret;
		}

		virtual random* getRandom() override final
		{
			return m_rnd;
		}

	private:
		random* m_rnd;
	};
}
