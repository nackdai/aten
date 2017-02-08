#pragma once

#include "types.h"

namespace aten {
	class sampler {
	public:
		sampler() {}
		virtual ~sampler() {}

		virtual real nextSample() = 0;
	};
}
