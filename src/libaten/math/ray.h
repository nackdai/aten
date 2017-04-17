#pragma once

#include "math/vec3.h"

namespace aten
{
	struct ray
	{
		ray() {}
		ray(const vec3& o, const vec3& d)
		{
			dir = normalize(d);
			//org = o;
			org = o + AT_MATH_EPSILON * dir;
		}

		vec3 org;
		vec3 dir;
	};
}
