#include "renderer/envmap.h"

namespace aten
{
	vec3 envmap::convertUVToDirection(real u, real v)
	{
		// u = phi / 2PI
		// => phi = 2PI * u;
		auto phi = 2 * AT_MATH_PI * u;

		// v = 1 - theta / PI
		// => theta = (1 - v) * PI;
		auto theta = (1 - v) * AT_MATH_PI;

		vec3 dir;

		dir.y = aten::cos(theta);

		auto xz = aten::sqrt(1 - dir.y * dir.y);

		dir.x = xz * aten::sin(phi);
		dir.z = xz * aten::cos(phi);

		// ”O‚Ì‚½‚ß...
		dir.normalize();

		return std::move(dir);
	}

	vec3 envmap::convertDirectionToUV(const vec3& dir)
	{
		auto temp = aten::atan2(dir.x, dir.z);
		auto r = dir.length();

		// Account for discontinuity
		auto phi = (real)((temp >= 0) ? temp : (temp + 2 * AT_MATH_PI));
		auto theta = aten::acos(dir.y / r);

		// Map to [0,1]x[0,1] range and reverse Y axis
		real u = phi / (2 * AT_MATH_PI);
		real v = 1 - theta / AT_MATH_PI;

		vec3 uv(u, v, 0);

		return std::move(uv);
	}

	vec3 envmap::sample(const ray& inRay) const
	{
		AT_ASSERT(m_envmap);

		// Translate cartesian coordinates to spherical system.
		const vec3& dir = inRay.dir;

		auto uv = convertDirectionToUV(dir);
		auto u = uv.x;
		auto v = uv.y;

		auto ret = m_envmap->at(u, v);

		return std::move(ret);
	}

	vec3 envmap::sample(real u, real v) const
	{
		auto ret = m_envmap->at(u, v);

		return std::move(ret);
	}
}
