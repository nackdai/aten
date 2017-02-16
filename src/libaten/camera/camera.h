#pragma once

#include "types.h"
#include "math/ray.h"

namespace aten {
	class camera {
	public:
		camera() {}
		virtual ~camera() {}

		virtual ray sample(real s, real t) = 0;
	};
}