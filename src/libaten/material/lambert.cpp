#include "material/lambert.h"

namespace aten {
	real lambert::pdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		return pdf(normal, wo);
	}

	real lambert::pdf(
		const vec3& normal,
		const vec3& wo)
	{
		auto c = dot(normal, wo);
		//AT_ASSERT(c >= 0);
		//c = aten::abs(c);

		auto ret = c / AT_MATH_PI;

		return ret;
	}

	real lambert::pdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto ret = pdf(m_param, normal, wi, wo, u, v);
		return ret;
	}

	vec3 lambert::sampleDirection(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		real u, real v,
		sampler* sampler)
	{
		return std::move(sampleDirection(normal, sampler));
	}

	vec3 lambert::sampleDirection(
		const vec3& normal,
		sampler* sampler)
	{
		// normalの方向を基準とした正規直交基底(w, u, v)を作る.
		// この基底に対する半球内で次のレイを飛ばす.
		vec3 n, t, b;

		n = normal;

		// nと平行にならないようにする.
		if (fabs(n.x) > AT_MATH_EPSILON) {
			t = normalize(cross(vec3(0.0, 1.0, 0.0), n));
		}
		else {
			t = normalize(cross(vec3(1.0, 0.0, 0.0), n));
		}
		b = cross(n, t);

		// コサイン項を使った重点的サンプリング.
		const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
		const real r2 = sampler->nextSample();
		const real r2s = sqrt(r2);

		const real x = aten::cos(r1) * r2s;
		const real y = aten::sin(r1) * r2s;
		const real z = aten::sqrt(real(1) - r2);

		vec3 dir = normalize((t * x + b * y + n * z));
		AT_ASSERT(dot(normal, dir) >= 0);

		return std::move(dir);
	}

	vec3 lambert::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		return std::move(sampleDirection(m_param, normal, ray.dir, u, v, sampler));
	}

	vec3 lambert::bsdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		return std::move(bsdf(param, u, v));
	}

	vec3 lambert::bsdf(
		const MaterialParameter& param,
		real u, real v)
	{
		vec3 albedo = param.baseColor;
		albedo *= sampleTexture(
			(texture*)param.albedoMap.ptr,
			u, v,
			real(1));

		vec3 ret = albedo / AT_MATH_PI;
		return ret;
	}

	vec3 lambert::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto ret = bsdf(m_param, normal, wi, wo, u, v);
		return std::move(ret);
	}

	MaterialSampling lambert::sample(
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

	MaterialSampling lambert::sample(
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