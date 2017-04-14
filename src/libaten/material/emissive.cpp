#include "material/emissive.h"
#include "material/lambert.h"

namespace aten {
	real emissive::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto ret = lambert::pdf(normal, wo);
		return ret;
	}

	vec3 emissive::sampleDirection(
		const ray& ray,
		const vec3& normal, 
		real u, real v,
		sampler* sampler) const
	{
		return std::move(lambert::sampleDirection(normal, sampler));
	}

	vec3 emissive::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto ret = lambert::bsdf((material*)this, u, v);
		return std::move(ret);
	}

	material::sampling emissive::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		sampling ret;

		const vec3& in = ray.dir;

		ret.dir = sampleDirection(ray, normal, u, v, sampler);
		ret.pdf = pdf(normal, in, ret.dir, u, v);
		ret.bsdf = bsdf(normal, in, ret.dir, u, v);

		return std::move(ret);
	}
}