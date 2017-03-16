#include "math/math.h"
#include "material/blinn.h"

namespace aten
{
	// NOTE
	// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-bsdf/

	real MicrofacetBlinn::pdf(
		const vec3& normal, 
		const vec3& wi,	/* in */
		const vec3& wo,	/* out */
		real u, real v,
		sampler* sampler) const
	{
		// NOTE
		// http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
		// Half-angle based

		// NOTE
		// https://segmentfault.com/a/1190000000432254

		// half vector.
		auto wh = normalize(-wi + wo);

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
		real u, real v,
		sampler* sampler) const
	{
		// NOTE
		// http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
		// Lobe Distribution Sampling

		// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-bsdf/
		// Sampling Blinn

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

#if 1
		// Sample halfway vector first, then reflect wi around that
		auto costheta = aten::pow(r1, 1 / (m_shininess + 1));
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		// phi = 2*PI*ksi2
		auto cosphi = aten::cos(AT_MATH_PI_2 * r2);
		auto sinphi = aten::sqrt(real(1) - cosphi * cosphi);
#else
		auto theta = aten::acos(aten::pow(r1, 1 / (m_shininess + 1)));
		auto phi = AT_MATH_PI_2 * r2;

		auto costheta = aten::cos(theta);
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		auto cosphi = aten::cos(phi);
		auto sinphi = aten::sqrt(1 - cosphi * cosphi);
#endif

		// Ortho normal base.
		auto n = normal;
		auto t = getOrthoVector(normal);
		auto b = normalize(cross(n, t));

		auto w = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
		w.normalize();

		auto dir = in - 2 * dot(in, w) * w;

		return std::move(dir);
	}

	vec3 MicrofacetBlinn::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		real fresnel = 1;
		auto ret = bsdf(fresnel, normal, wi, wo, u, v);
		return std::move(ret);
	}

	vec3 MicrofacetBlinn::bsdf(
		real& fresnel,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
		real ni = real(1);	// ^‹ó

		// •¨‘Ì“à•”‚Ì‹üÜ—¦.
		real nt = ior();

		vec3 V = -wi;
		vec3 L = wo;
		vec3 N = normal;
		vec3 H = normalize(L + V);

		// TODO
		// Desney‚¾‚Æabs‚µ‚Ä‚È‚¢‚ªAAMD‚Ì‚Í‚µ‚Ä‚¢‚é....
		auto NdotH = aten::abs(dot(N, H));
		auto VdotH = aten::abs(dot(V, H));
		auto NdotL = aten::abs(dot(N, L));
		auto NdotV = aten::abs(dot(N, V));

		auto a = m_shininess;

		real F(1);
		{
			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) à R0 + (1 - R0)(1 - cosƒ¦)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (ni - nt) / (ni + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(L, H));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		auto denom = 4 * NdotL * NdotV;

		// Compute D.
#if 0
		// NOTE
		// https://www.siggraph.org/education/materials/HyperGraph/illumin/specular_highlights/blinn_model_for_specular_reflect_1.htm
		auto x = aten::acos(NdotH) * a;
		real D = aten::exp(-x * x);
#else
		// NOTE
		// http://simonstechblog.blogspot.jp/2011/12/microfacet-bsdf.html
		real D = (a + 2) / (2 * AT_MATH_PI);
		D *= aten::pow(std::max((real)0, NdotH), a);
#endif

		// Compute G.
		real G(1);
		{
			// Cook-Torrance geometry function.
			// http://simonstechblog.blogspot.jp/2011/12/microfacet-bsdf.html

			auto G1 = 2 * NdotH * NdotL / VdotH;
			auto G2 = 2 * NdotH * NdotV / VdotH;
			G = std::min((real)1, std::min(G1, G2));
		}

		auto albedo = color();
		albedo *= sampleAlbedoMap(u, v);

		auto bsdf = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : 0;

		fresnel = F;

		return std::move(bsdf);
	}

	material::sampling MicrofacetBlinn::sample(
		const vec3& in,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		ret.dir = sampleDirection(in, normal, u, v, sampler);

#if 1
		ret.pdf = pdf(normal, in, ret.dir, u, v, sampler);

		real fresnel = 1;
		ret.bsdf = bsdf(fresnel, normal, in, ret.dir, u, v);
		ret.fresnel = fresnel;
#else
		vec3 V = -in;
		vec3 L = ret.dir;
		vec3 N = normal;
		vec3 H = normalize(L + V);

		auto NdotL = aten::abs(dot(N, L));
		auto NdotH = aten::abs(dot(N, H));

		if (NdotL > 0 && NdotH > 0) {
			ret.pdf = pdf(normal, in, ret.dir, u, v);

			ret.bsdf = bsdf(normal, in, ret.dir, u, v);
		}
		else {
			ret.pdf = 0;
		}
#endif

		return std::move(ret);
	}
}
