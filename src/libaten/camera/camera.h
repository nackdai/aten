#pragma once

#include "types.h"
#include "renderer/ray.h"

namespace aten {
	class camera {
	public:
		camera() {}
		virtual ~camera() {}

		virtual ray sample(real s, real t) = 0;
	};
}