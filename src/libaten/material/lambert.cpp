#include "material/lambert.h"

namespace AT_NAME {
	AT_DEVICE_API real lambert::pdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		return pdf(normal, wo);
	}

	AT_DEVICE_API real lambert::pdf(
		const aten::vec3& normal,
		const aten::vec3& wo)
	{
		auto c = dot(normal, wo);
		//AT_ASSERT(c >= 0);
		//c = aten::abs(c);

		auto ret = c / AT_MATH_PI;

		return ret;
	}

	real lambert::pdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		auto ret = pdf(&m_param, normal, wi, wo, u, v);
		return ret;
	}

	AT_DEVICE_API aten::vec3 lambert::sampleDirection(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		return std::move(sampleDirection(normal, sampler));
	}

	AT_DEVICE_API aten::vec3 lambert::sampleDirection(
		const aten::vec3& normal,
		aten::sampler* sampler)
	{
		// normalの方向を基準とした正規直交基底(w, u, v)を作る.
		// この基底に対する半球内で次のレイを飛ばす.
#if 1
		auto n = normal;
		auto t = getOrthoVector(n);
		auto b = cross(n, t);
#else
		aten::vec3 n, t, b;

		n = normal;

		// nと平行にならないようにする.
		if (fabs(n.x) > AT_MATH_EPSILON) {
			t = normalize(cross(aten::make_float3(0.0, 1.0, 0.0), n));
		}
		else {
			t = normalize(cross(aten::make_float3(1.0, 0.0, 0.0), n));
		}
		b = cross(n, t);
#endif

		// コサイン項を使った重点的サンプリング.
		const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
		const real r2 = sampler->nextSample();
		const real r2s = sqrt(r2);

		const real x = aten::cos(r1) * r2s;
		const real y = aten::sin(r1) * r2s;
		const real z = aten::sqrt(real(1) - r2);

		aten::vec3 dir = normalize((t * x + b * y + n * z));
		AT_ASSERT(dot(normal, dir) >= 0);

		return std::move(dir);
	}

	aten::vec3 lambert::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
	}

	AT_DEVICE_API aten::vec3 lambert::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		return std::move(bsdf(param, u, v));
	}

	AT_DEVICE_API aten::vec3 lambert::bsdf(
		const aten::MaterialParameter* param,
		real u, real v)
	{
		aten::vec3 albedo = param->baseColor;
		albedo *= sampleTexture(
			(aten::texture*)param->albedoMap.ptr,
			u, v,
			real(1));

		aten::vec3 ret = albedo / AT_MATH_PI;
		return ret;
	}

	aten::vec3 lambert::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		auto ret = bsdf(&m_param, normal, wi, wo, u, v);
		return std::move(ret);
	}

	AT_DEVICE_API void lambert::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		MaterialSampling ret;

		result->dir = sampleDirection(param, normal, wi, u, v, sampler);
		result->pdf = pdf(param, normal, wi, result->dir, u, v);
		result->bsdf = bsdf(param, normal, wi, result->dir, u, v);
	}

	MaterialSampling lambert::sample(
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