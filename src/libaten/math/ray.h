#pragma once

#include "math/vec3.h"

namespace aten
{
	struct ray
	{
		AT_DEVICE_API ray()
		{
			isActive = true;
		}
		AT_DEVICE_API ray(const vec3& o, const vec3& d)
		{
			dir = normalize(d);
			org = o + AT_MATH_EPSILON * dir;

			isActive = true;
		}

		vec3 org;
		vec3 dir;
		struct {
			uint32_t isActive : 1;
		};
	};
}
