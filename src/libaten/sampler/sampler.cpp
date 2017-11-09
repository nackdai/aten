#include <random>
#include <algorithm>
#include "sampler/samplerinterface.h"
#include "sampler/halton.h"

namespace aten {
	static std::vector<uint32_t> g_random;

	void initSampler(
		uint32_t width, uint32_t height, 
		int seed/*= 0*/,
		bool needInitHalton/*= false*/)
	{
		// TODO
		::srand(seed);

		if (needInitHalton) {
			Halton::makePrimeNumbers();
		}

		g_random.resize(width * height);
		std::mt19937 rand_src(seed);
		std::generate(g_random.begin(), g_random.end(), rand_src);
	}

	const std::vector<uint32_t>& getRandom()
	{
		return g_random;
	}
	uint32_t getRandom(uint32_t idx)
	{
		return g_random[idx % g_random.size()];
	}
}
