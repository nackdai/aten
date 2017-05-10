#include "material/emissive.h"
#include "material/lambert.h"

namespace AT_NAME {
	AT_DEVICE_API real emissive::pdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto ret = lambert::pdf(param, normal, wi, wo, u, v);
		return ret;
	}

	real emissive::pdf(
		const aten::vec3& normal, 
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return emissive::pdf(&m_param, normal, wi, wo, u, v);
	}

	AT_DEVICE_API aten::vec3 emissive::sampleDirection(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		return std::move(lambert::sampleDirection(param, normal, wi, u, v, sampler));
	}

	aten::vec3 emissive::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal, 
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(emissive::sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
	}

	AT_DEVICE_API aten::vec3 emissive::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto ret = lambert::bsdf(param, normal, wi, wo, u, v);
		return std::move(ret);
	}

	aten::vec3 emissive::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(emissive::bsdf(&m_param, normal, wi, wo, u, v));
	}

	AT_DEVICE_API void emissive::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		result->dir = sampleDirection(param, normal, wi, u, v, sampler);
		result->pdf = pdf(param, normal, wi, result->dir, u, v);
		result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
	}

	MaterialSampling emissive::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		MaterialSampling ret;
		
		sample(
			&ret,
			&m_param,
			normal,
			ray.dir,
			orgnormal,
			sampler,
			u, v,
			isLightPath);

		return std::move(ret);
	}
}