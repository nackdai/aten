#include "material/blinn.h"

namespace aten
{
	// NOTE
	// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

	real MicrofacetBlinn::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
	{
		// NOTE
		// http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
		// Half-angle based

		// half vector.
		auto wh = normalize(wi + wo);

		auto costheta = dot(normal, wh);

		auto n = m_shininess;

		auto c = dot(wo, wh);

		real pdf = c > AT_MATH_EPSILON ?
			((n + 1) / (2 * AT_MATH_PI)) * (aten::pow(costheta, n) / (4 * c))
			: 0;

		return pdf;
	}

	vec3 MicrofacetBlinn::sampleDirection(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		// NOTE
		// http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
		// Lobe Distribution Sampling

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		// Sample halfway vector first, then reflect wi around that
		auto costheta = aten::pow(r1, 1 / (m_shininess + 1));
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		// phi = 2*PI*ksi2
		auto cosphi = aten::cos(AT_MATH_PI_2 * r2);
		auto sinphi = aten::sqrt(real(1) - cosphi * cosphi);

		// Ortho normal base.
		auto w = normal;
		auto u = getOrthoVector(normal);
		auto v = normalize(cross(w, u));

		auto dir = u * sintheta * cosphi + v * sintheta * sinphi * w * costheta;

		return std::move(dir);
	}

	vec3 MicrofacetBlinn::brdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
		real ni = real(1);	// ^‹ó

		// •¨‘Ì“à•”‚Ì‹üÜ—¦.
		real nt = m_nt;

		auto n = normal;

		auto a = m_shininess;

		auto F = computFresnel(wi, normal, ni, nt);

		// Incident and reflected zenith angles
		auto costhetao = dot(normal, wo);
		auto costhetai = dot(normal, wi);

		auto denom = 4 * costhetao * costhetai;

		auto wh = normalize(wi + wo);

		// Compute D.
		real D(1);
		{
			auto c = dot(normal, wh);
			D = ((a + 2) * aten::pow(c, a)) / (2 * AT_MATH_PI);
		}

		// Compute G.
		real G(1);
		{
			auto ndotwh = aten::abs(dot(n, wh));
			auto ndotwo = aten::abs(dot(n, wo));
			auto ndotwi = aten::abs(dot(n, wi));
			auto wodotwh = aten::abs(dot(wo, wh));

			G = min(
				1, 
				min(2 * ndotwh * ndotwo / wodotwh, 2.f * ndotwh * ndotwi / wodotwh));
		}

		auto albedo = m_color;
		if (m_tex) {
			auto texclr = m_tex->at(u, v);
			albedo *= texclr;
		}

		auto brdf = albedo * F * G * D / denom;

		return std::move(brdf);
	}

	material::sampling MicrofacetBlinn::sample(
		const vec3& in,
		const vec3& normal,
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
