#include "material/emissive.h"
#include "material/lambert.h"

namespace aten {
	real emissive::pdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		auto ret = lambert::pdf(param, normal, wi, wo, u, v);
		return ret;
	}

	real emissive::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return emissive::pdf(m_param, normal, wi, wo, u, v);
	}

	vec3 emissive::sampleDirection(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		real u, real v,
		sampler* sampler)
	{
		return std::move(lambert::sampleDirection(param, normal, wi, u, v, sampler));
	}

	vec3 emissive::sampleDirection(
		const ray& ray,
		const vec3& normal, 
		real u, real v,
		sampler* sampler) const
	{
		return std::move(emissive::sampleDirection(m_param, normal, ray.dir, u, v, sampler));
	}

	vec3 emissive::bsdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		auto ret = lambert::bsdf(param, normal, wi, wo, u, v);
		return std::move(ret);
	}

	vec3 emissive::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return std::move(emissive::bsdf(m_param, normal, wi, wo, u, v));
	}

	MaterialSampling emissive::sample(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		MaterialSampling ret;

		ret.dir = sampleDirection(param, normal, wi, u, v, sampler);
		ret.pdf = pdf(param, normal, wi, ret.dir, u, v);
		ret.bsdf = bsdf(param, normal, wi, ret.dir, u, v);

		return std::move(ret);
	}

	MaterialSampling emissive::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		auto ret = sample(
			m_param,
			normal,
			ray.dir,
			hitrec,
			sampler,
			u, v,
			isLightPath);

		return std::move(ret);
	}
}