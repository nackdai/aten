#pragma once

#include "types.h"

namespace aten {
	class random {
	public:
		random() {}
		virtual ~random() {}

		// [0, 1]
		virtual real next01() = 0;
	};

	inline aten::real drand48()
	{
		return (aten::real)::rand() / RAND_MAX;
	}
}
