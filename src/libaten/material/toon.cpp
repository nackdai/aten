#include "material/toon.h"

namespace aten
{
	vec3 toon::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		auto* light = getTargetLight();
		AT_ASSERT(light);

		auto res = light->sample(ray.org, sampler);
		
		return std::move(res.dir);
	}

	vec3 toon::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		real cosShadow = dot(normal, wo);
		auto ret = bsdf(cosShadow, u, v);
		return std::move(ret);
	}

	material::sampling toon::sample(
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

		ret.fresnel = 1;

		return std::move(ret);
	}

	vec3 toon::bsdf(
		real cosShadow,
		real u, real v) const
	{
		const real c = aten::clamp<real>(cosShadow, 0, 1);

		vec3 albedo = color();
		albedo *= sampleAlbedoMap(u, v);

		real coeff = 1;

		if (m_func) {
			coeff = m_func(c);
		}
		else {
			// “K“–...
			if (c < 0.5) {
				coeff = 0;
			}
			else {
				coeff = 1;
			}
		}

		// TODO
		// Š®‘S‚É‰e‚É‚È‚ç‚È‚¢‚æ‚¤‚É‚µ‚Ä‚Ý‚é.
		coeff = std::max<real>(coeff, real(0.05));
		//coeff = aten::clamp<real>(coeff, 0.05, 1);

		vec3 bsdf = albedo * coeff;

		return std::move(bsdf);
	}
}
