#pragma once

#include "defs.h"
#include "types.h"

namespace aten {
	class sampler {
	public:
		static void init();

	public:
		sampler() {}
		virtual ~sampler() {}

		virtual void init(uint32_t seed)
		{
			// Nothing is done...
		}
		virtual AT_DEVICE_API real nextSample() = 0;
	};

	inline real drand48()
	{
		return (real)::rand() / RAND_MAX;
	}
}
