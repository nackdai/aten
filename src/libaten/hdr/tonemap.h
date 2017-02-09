#pragma once

#include "math/vec3.h"
#include "misc/color.h"

namespace aten
{
	class Tonemap {
	public:
		static void doTonemap(
			int width, int height,
			const vec3* src,
			color* dst);
	};
}