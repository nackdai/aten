#pragma once

#include "material/specular.h"

namespace aten
{
	real specular::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
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

	vec3 specular::bsdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto c = dot(normal, wo);

		vec3 bsdf;

		if (c > 0) {
			bsdf = m_color / c;
		}

		return std::move(bsdf);
	}

	material::sampling specular::sample(
		const vec3& in,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		ret.dir = sampleDirection(in, normal, sampler);
		ret.pdf = pdf(normal, in, ret.dir);
		ret.bsdf = bsdf(normal, in, ret.dir, u, v);

		return std::move(ret);
	}
}
