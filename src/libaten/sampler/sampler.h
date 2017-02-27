#pragma once

#include "types.h"

namespace aten {
	class random;

	class sampler {
	public:
		sampler() {}
		virtual ~sampler() {}

		virtual real nextSample() = 0;

		virtual random* getRandom()
		{
			return nullptr;
		}
	};
}
