#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
	// Wang's hash による乱数ジェネレータ.
	class WangHash AT_INHERIT(sampler) {
	public:
		WangHash() {}
		WangHash(const unsigned int initial_seed)
		{
			init(initial_seed);
		}

		AT_VIRTUAL(~WangHash() {})

		AT_VIRTUAL_OVERRIDE_FINAL(void init(uint32_t seed))
		{
			m_seed = seed;
		}

		// [0, 1]
		AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
		{
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

		unsigned int m_seed;
	};
}
