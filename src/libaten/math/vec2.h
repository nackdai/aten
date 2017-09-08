#pragma once

#include "defs.h"
#include "math/math.h"

namespace aten {
	struct vec2 {
		real x;
		real y;

		AT_DEVICE_API vec2()
		{
			x = y = 0;
		}
		AT_DEVICE_API vec2(const vec2& _v)
		{
			x = _v.x;
			y = _v.y;
		}
		AT_DEVICE_API vec2(real f)
		{
			x = y = f;
		}
		AT_DEVICE_API vec2(real _x, real _y)
		{
			x = _x;
			y = _y;
		}
	};
}
