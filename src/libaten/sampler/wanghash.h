#pragma once

#include "types.h"
#include "sampler/sampler.h"

namespace aten {
	// Wang's hash による乱数ジェネレータ.
	class WangHash : public sampler {
	public:
		WangHash() {}
		WangHash(const unsigned int initial_seed)
		{
			init(initial_seed);
		}

		virtual ~WangHash() {}

		virtual void init(uint32_t seed) override final
		{
			m_param.seed = seed;
		}

		// [0, 1]
		virtual real nextSample() override final
		{
			return nextSample(&m_param);
		}

		static AT_DEVICE_API real nextSample(SamplerParameter* param)
		{
			return (real)next(param) / UINT_MAX;
		}

	private:
		// NOTE
		// https://gist.github.com/badboy/6267743
		static uint32_t next(SamplerParameter* param)
		{
			uint32_t key = 1664525U * param->seed + 1013904223U;

			key = (key ^ 61) ^ (key >> 16);
			key = key + (key << 3);
			key = key ^ (key >> 4);
			key = key * 0x27d4eb2d;
			key = key ^ (key >> 15);

			param->seed += 1;

			return key;
		}

		SamplerParameter m_param;
	};
}
