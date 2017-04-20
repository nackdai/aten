#pragma once

#include "material/specular.h"

namespace aten
{
	real specular::pdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		return real(1);
	}

	real specular::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return pdf(m_param, normal, wi, wo, u, v);
	}

	vec3 specular::sampleDirection(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		real u, real v,
		sampler* sampler)
	{
		auto reflect = wi - 2 * dot(normal, wi) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	vec3 specular::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		const vec3& in = ray.dir;

		return std::move(sampleDirection(m_param, normal, in, u, v, sampler));
	}

	vec3 specular::bsdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		auto c = dot(normal, wo);

#if 1
		vec3 bsdf = param.baseColor;
#else
		vec3 bsdf;

		// For canceling cosine factor.
		if (c > 0) {
			bsdf = m_color / c;
		}
#endif

		bsdf *= sampleTexture((texture*)param.albedoMap.ptr, u, v, real(1));

		return std::move(bsdf);
	}

	vec3 specular::bsdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(m_param, normal, wi, wo, u, v));
	}

	MaterialSampling specular::sample(
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

	MaterialSampling specular::sample(
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
}
