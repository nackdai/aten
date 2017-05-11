#pragma once

#include "types.h"
#include "sampler/sampler.h"

namespace aten {
	// Wang's hash による乱数ジェネレータ.
	class WangHash : public sampler {
		unsigned int m_seed{ 0 };

	public:
		WangHash() {}
		WangHash(const unsigned int initial_seed)
		{
			init(initial_seed);
		}

		virtual ~WangHash() {}

		virtual void init(uint32_t seed) override final
		{
			m_seed = seed;
		}

		// [0, 1]
		virtual real nextSample() override final
		{
			//return (real)next() / (UINT_MAX + 1.0);
			return (real)next() / UINT_MAX;
		}

	private:
		// NOTE
		// https://gist.github.com/badboy/6267743
		uint32_t next()
		{
			uint32_t key = 1664525U * m_seed + 1013904223U;

			key = (key ^ 61) ^ (key >> 16);
			key = key + (key << 3);
			key = key ^ (key >> 4);
			key = key * 0x27d4eb2d;
			key = key ^ (key >> 15);

			m_seed += 1;

			return key;
		}
	};
}
