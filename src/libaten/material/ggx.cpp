#include "material/ggx.h"

namespace aten
{
	real MicrofacetGGX::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
	{
	}

	vec3 MicrofacetGGX::sampleDirection(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();
	}

	vec3 MicrofacetGGX::brdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// Compute D.
		real D(1);
		{

		}

		// Compute G.
		real G(1);
		{
			// NOTE
			// http://computergraphics.stackexchange.com/questions/2489/correct-form-of-the-ggx-geometry-term
		}

		auto albedo = m_color;
		if (m_tex) {
			auto texclr = m_tex->at(u, v);
			albedo *= texclr;
		}

		auto brdf = albedo * F * G * D / denom;
		//auto brdf = albedo * G * D / denom;

		return std::move(brdf);
	}

	material::sampling MicrofacetGGX::sample(
		const vec3& in,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		ret.dir = sampleDirection(in, normal, sampler);
		ret.pdf = pdf(normal, in, ret.dir);

		ret.brdf = brdf(normal, in, ret.dir, u, v);

		return std::move(ret);
	}
}
