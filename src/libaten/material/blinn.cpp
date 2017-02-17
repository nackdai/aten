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
		sampler* sampler) const
	{
		// NOTE
		// http://digibug.ugr.es/bitstream/10481/19751/1/rmontes_LSI-2012-001TR.pdf
		// Lobe Distribution Sampling

		// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
		// Sampling Blinn

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		// Sample halfway vector first, then reflect wi around that
		auto costheta = aten::pow(r1, 1 / (m_shininess + 1));
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		// phi = 2*PI*ksi2
		auto cosphi = aten::cos(AT_MATH_PI_2 * r2);
		auto sinphi = aten::sqrt(real(1) - cosphi * cosphi);

		// Ortho normal base.
		auto n = normal;
#if 0
		auto u = getOrthoVector(normal);
		auto v = normalize(cross(n, u));
#else
		vec3 u, v;

		// wと平行にならないようにする.
		if (fabs(n.x) > 0.1) {
			u = normalize(cross(vec3(0.0, 1.0, 0.0), n));
		}
		else {
			u = normalize(cross(vec3(1.0, 0.0, 0.0), n));
		}
		v = cross(n, u);
#endif

		auto w = u * sintheta * cosphi + v * sintheta * sinphi + n * costheta;
		w.normalize();

		auto dir = in - 2 * dot(in, w) * w;

		return std::move(dir);
	}

	vec3 MicrofacetBlinn::brdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// レイが入射してくる側の物体の屈折率.
		real ni = real(1);	// 真空

		// 物体内部の屈折率.
		real nt = m_nt;

		vec3 V = -wi;
		vec3 L = wo;
		vec3 N = normal;
		vec3 H = normalize(L + V);

		// TODO
		// Desneyだとabsしてないが、AMDのはしている....
		auto NdotH = aten::abs(dot(N, H));
		auto VdotH = aten::abs(dot(V, H));
		auto NdotL = aten::abs(dot(N, L));
		auto NdotV = aten::abs(dot(N, V));

		auto n = m_shininess;

		auto F = computFresnel(wi, normal, ni, nt);

		auto denom = 4 * NdotL * NdotV;

		// Compute D.
		real D(1);
		{
			auto x = aten::acos(NdotH) * n;
			D = aten::exp(-x * x);
		}

		// Compute G.
		real G(1);
		{
			auto G1 = 2 * NdotH * NdotL / VdotH;
			auto G2 = 2 * NdotH * NdotV / VdotH;
			G = min(1, min(G1, G2));
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

	material::sampling MicrofacetBlinn::sample(
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
