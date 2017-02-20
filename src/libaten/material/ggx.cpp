#include "material/ggx.h"

namespace aten
{
	// NOTE
	// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

	real sampleGGX_D(
		const vec3& wh,	// half
		const vec3& n,	// normal
		real roughness)
	{
		// NOTE
		// ((a^2 - 1) * cos^2 + 1)^2
		// (-> a^2 = a2, cos^2 = cos2)
		// ((a2 - 1) * cos2 + 1)^2
		//  = (a2cos2 + 1 - cos2)^2 = (a2cos2 + sin2)^2
		// (-> sin = sqrt(1 - cos2), sin2 = 1 - cos2)
		//  = a4 * cos4 + 2 * a2 * cos2 * sin2 + sin4
		//  = cos4 * (a4 + 2 * a2 * (sin2 / cos2) + (sin4 / cos4))
		//  = cos4 * (a4 + 2 * a2 * tan2 + tan4)
		//  = cos4 * (a2 + tan2) ^ 2

		real a = roughness;
		auto a2 = a * a;

		auto costheta = aten::abs(dot(wh, n));
		auto cos2 = costheta * costheta;

		auto denom = aten::pow((a2 - 1) * cos2 + 1, 2);

		auto D = denom > 0 ? a2 / (AT_MATH_PI * denom) : 0;

		return D;
	}

	real MicrofacetGGX::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
	{
		auto wh = normalize(-wi + wo);

		auto costheta = aten::abs(dot(wh, normal));

		auto D = sampleGGX_D(wh, normal, m_roughness);

		auto denom = 4 * aten::abs(dot(wo, wh));

		auto pdf = denom > 0 ? (D * costheta) / denom : 0;

		return pdf;
	}

	vec3 MicrofacetGGX::sampleDirection(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		auto a = m_roughness;

		auto theta = aten::atan(a * aten::sqrt(r1 / (1 - r1)));
		theta = ((theta >= 0) ? theta : (theta + 2 * AT_MATH_PI));

		auto phi = 2 * AT_MATH_PI * r2;

		auto costheta = aten::cos(theta);
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		auto cosphi = aten::cos(phi);
		auto sinphi = aten::sqrt(1 - cosphi * cosphi);

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

	vec3 MicrofacetGGX::brdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// レイが入射してくる側の物体の屈折率.
		real ni = real(1);	// 真空

							// 物体内部の屈折率.
		real nt = m_ior;

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

		// Compute D.
		real D = sampleGGX_D(H, normal, m_roughness);

		// Compute G.
		real G(1);
		{
			// NOTE
			// http://computergraphics.stackexchange.com/questions/2489/correct-form-of-the-ggx-geometry-term
			// http://gregory-igehy.hatenadiary.com/entry/2015/02/26/154142
			


		}

		auto albedo = m_color;
		if (m_tex) {
			auto texclr = m_tex->at(u, v);
			albedo *= texclr;
		}

		auto F = computFresnel(wi, normal, ni, nt);

		auto denom = 4 * NdotL * NdotV;

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
