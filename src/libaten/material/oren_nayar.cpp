#include "material/oren_nayar.h"

namespace aten {
	// NOTE
	// https://ja.wikipedia.org/wiki/%E3%82%AA%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%BB%E3%83%8D%E3%82%A4%E3%83%A4%E3%83%BC%E5%8F%8D%E5%B0%84
	// https://github.com/imageworks/OpenShadingLanguage/blob/master/src/testrender/shading.cpp

	real OrenNayar::pdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		auto NL = dot(normal, wo);
		auto NV = dot(normal, -wi);

		real pdf = 0;

		if (NL > 0 && NV > 0) {
			pdf = NL / AT_MATH_PI;
		}

		return pdf;
	}

	real OrenNayar::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return pdf(m_param, normal, wi, wo, u, v);
	}

	vec3 OrenNayar::sampleDirection(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		real u, real v,
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

	vec3 OrenNayar::sampleDirection(
		const ray& ray,
		const vec3& normal, 
		real u, real v,
		sampler* sampler) const
	{
		return std::move(sampleDirection(m_param, normal, ray.dir, u, v, sampler));
	}

	vec3 OrenNayar::bsdf(
		const MaterialParameter& param,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v)
	{
		// NOTE
		// A tiny improvement of Oren-Nayar reflectance model
		// http://mimosa-pudica.net/improved-oren-nayar.html

		vec3 bsdf(0);

		const auto NL = dot(normal, wo);
		const auto NV = dot(normal, -wi);

		if (NL > 0 && NV > 0) {
			auto roughness = material::sampleTexture((texture*)param.roughnessMap.ptr, u, v, param.roughness);

			auto albedo = param.baseColor;
			albedo *= material::sampleTexture((texture*)param.albedoMap.ptr, u, v, real(1));

			const real a = roughness.r;
			const real a2 = a * a;

			const real A = real(1) - real(0.5) * (a2 / (a2 + real(0.33)));
			const real B = real(0.45) * (a2 / (a2 + real(0.09)));

			const auto LV = dot(wo, -wi);

			const auto s = LV - NL * NV;
			const auto t = s <= 0 ? 1 : std::max(NL, NV);

			bsdf = (albedo / AT_MATH_PI) * (A + B * (s / t));
		}

		return bsdf;
	}

	vec3 OrenNayar::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(m_param, normal, wi, wo, u, v));
	}

	MaterialSampling OrenNayar::sample(
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

	MaterialSampling OrenNayar::sample(
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