#pragma once

#include "math/vec3.h"

namespace aten
{
	class ray
	{
	public:
		ray() {}
		ray(const vec3& o, const vec3& d)
		{
			org = o;
			dir = normalize(d);
		}

		vec3 org;
		vec3 dir;
	};
}
