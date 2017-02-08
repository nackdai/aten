#pragma once

#include "types.h"

namespace aten {
	// Xor-Shiftによる乱数ジェネレータ.
	class XorShift {
		unsigned int seed_[4];

	public:
		XorShift() {}
		XorShift(const unsigned int initial_seed)
		{
			init(initial_seed);
		}

		~XorShift() {}

		unsigned int next(void)
		{
			const unsigned int t = seed_[0] ^ (seed_[0] << 11);
			seed_[0] = seed_[1];
			seed_[1] = seed_[2];
			seed_[2] = seed_[3];
			return seed_[3] = (seed_[3] ^ (seed_[3] >> 19)) ^ (t ^ (t >> 8));
		}

		// [0, 1]
		real next01(void)
		{
			//return (real)next() / (UINT_MAX + 1.0);
			return (real)next() / UINT_MAX;
		}

		void init(const unsigned int initial_seed)
		{
			unsigned int s = initial_seed;
			for (int i = 1; i <= 4; i++) {
				seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
			}
		}
	};
}
