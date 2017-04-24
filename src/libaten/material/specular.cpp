#pragma once

#include "material/specular.h"

namespace AT_NAME
{
	real specular::pdf(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		return real(1);
	}

	real specular::pdf(
		const aten::vec3& normal, 
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return pdf(m_param, normal, wi, wo, u, v);
	}

	aten::vec3 specular::sampleDirection(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		auto reflect = wi - 2 * dot(normal, wi) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	aten::vec3 specular::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		const aten::vec3& in = ray.dir;

		return std::move(sampleDirection(m_param, normal, in, u, v, sampler));
	}

	aten::vec3 specular::bsdf(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto c = dot(normal, wo);

#if 1
		aten::vec3 bsdf = param.baseColor;
#else
		aten::vec3 bsdf;

		// For canceling cosine factor.
		if (c > 0) {
			bsdf = m_color / c;
		}
#endif

		bsdf *= sampleTexture((aten::texture*)param.albedoMap.ptr, u, v, real(1));

		return std::move(bsdf);
	}

	aten::vec3 specular::bsdf(
		const aten::vec3& normal, 
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(m_param, normal, wi, wo, u, v));
	}

	MaterialSampling specular::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::hitrecord& hitrec,
		aten::sampler* sampler,
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
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::hitrecord& hitrec,
		aten::sampler* sampler,
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
