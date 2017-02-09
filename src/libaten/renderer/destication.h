#pragma once

#include "types.h"
#include "math/vec3.h"

namespace aten
{
	struct Destination {
		int width{ 0 };
		int height{ 0 };
		uint32_t maxDepth{ 1 };
		uint32_t sample{ 1 };
		uint32_t mutation{ 1 };
		vec3* buffer{ nullptr };
	};
}
