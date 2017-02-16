#pragma once

#include "material/specular.h"

namespace aten
{
	real specular::pdf(const vec3& normal, const vec3& dir) const
	{
		return real(1);
	}

	vec3 specular::sampleDirection(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		auto reflect = in - 2 * dot(normal, in) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	vec3 specular::brdf(
		const vec3& normal, 
		const vec3& dir,
		real u, real v) const
	{
		auto c = dot(normal, dir);

		vec3 brdf;

		if (c > 0) {
			brdf = m_color / c;
		}

		return std::move(brdf);
	}

	material::sampling specular::sample(
		const vec3& in,
		const vec3& normal,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		ret.dir = sampleDirection(in, normal, sampler);
		ret.pdf = pdf(normal, ret.dir);
		ret.brdf = brdf(normal, ret.dir, u, v);

		return std::move(ret);
	}
}
