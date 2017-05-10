#pragma once

#include "types.h"
#include "sampler/random.h"

namespace aten {
	// Xor-Shiftによる乱数ジェネレータ.
	class WangHash : public random {
		unsigned int seed;

	public:
		WangHash() {}
		WangHash(const unsigned int initial_seed)
		{
			seed = initial_seed;
		}

		virtual ~WangHash() {}

		// [0, 1]
		virtual real next01() override final
		{
			//return (real)next() / (UINT_MAX + 1.0);
			return (real)next() / UINT_MAX;
		}

	private:
		// NOTE
		// https://gist.github.com/badboy/6267743
		uint32_t next()
		{
			uint32_t key = 1664525U * seed + 1013904223U;

			key = (key ^ 61) ^ (key >> 16);
			key = key + (key << 3);
			key = key ^ (key >> 4);
			key = key * 0x27d4eb2d;
			key = key ^ (key >> 15);

			seed += 1;

			return key;
		}
	};
}
