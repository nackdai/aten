#include "renderer/envmap.h"

namespace aten
{
	vec3 envmap::sample(const ray& inRay) const
	{
		AT_ASSERT(m_envmap);

		// Translate cartesian coordinates to spherical system.
		const vec3& dir = inRay.dir;

		auto temp = aten::atan2(dir.x, dir.z);
		auto r = dir.length();

		// Account for discontinuity
		auto phi = (real)((temp >= 0) ? temp : (temp + 2 * AT_MATH_PI));
		auto theta = aten::acos(dir.y / r);

		// Map to [0,1]x[0,1] range and reverse Y axis
		real u = phi / (2 * AT_MATH_PI);
		real v = 1 - theta / AT_MATH_PI;

		auto ret = m_envmap->at(u, v);

		return std::move(ret);
	}
}
