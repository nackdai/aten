#pragma once

#include "material/specular.h"

namespace aten
{
	real specular::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return real(1);
	}

	vec3 specular::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		const vec3& in = ray.dir;

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

#if 1
		vec3 bsdf = color();
#else
		vec3 bsdf;

		// For canceling cosine factor.
		if (c > 0) {
			bsdf = m_color / c;
		}
#endif

		bsdf *= sampleAlbedoMap(u, v);

		return std::move(bsdf);
	}

	material::sampling specular::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		const vec3& in = ray.dir;

		ret.dir = sampleDirection(ray, normal, u, v, sampler);
		ret.pdf = pdf(normal, in, ret.dir, u, v);
		ret.bsdf = bsdf(normal, in, ret.dir, u, v);

		return std::move(ret);
	}
}
