#pragma once

#include <vector>
#include "defs.h"
#include "types.h"

namespace aten {
#ifndef __AT_CUDA__
	class sampler {
	public:
		sampler() {}
		virtual ~sampler() {}

		virtual void init(uint32_t seed, const unsigned int* data = nullptr)
		{
			// Nothing is done...
		}
		virtual AT_DEVICE_API real nextSample() = 0;
	};
#endif

	void initSampler(
		uint32_t width, uint32_t height, 
		int seed = 0,
		bool needInitHalton = false);

	const std::vector<uint32_t>& getRandom();
	uint32_t getRandom(uint32_t idx);

	inline real drand48()
	{
		return (real)::rand() / RAND_MAX;
	}
}
