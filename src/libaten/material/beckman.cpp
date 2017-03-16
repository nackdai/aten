#include "material/beckman.h"

namespace aten
{
	real MicrofacetBeckman::sampleRoughness(real u, real v) const
	{
		vec3 roughness = material::sampleTexture(m_roughnessMap, u, v, m_roughness);
		return roughness.r;
	}

	real MicrofacetBeckman::pdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v,
		sampler* sampler) const
	{
		auto roughness = sampleRoughness(u, v);
		auto ret = pdf(roughness, normal, wi, wo);
		return ret;
	}

	vec3 MicrofacetBeckman::sampleDirection(
		const vec3& in,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		auto roughness = sampleRoughness(u, v);
		vec3 dir = sampleDirection(roughness, in, normal, sampler);
		return std::move(dir);
	}

	vec3 MicrofacetBeckman::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto roughness = sampleRoughness(u, v);
		real fresnel = 1;
		vec3 ret = bsdf(roughness, fresnel, normal, wi, wo, u, v);
		return std::move(ret);
	}

	real sampleBeckman_D(
		const vec3& wh,	// half
		const vec3& n,	// normal
		real roughness)
	{
		// NOTE
		// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-bsdf/

		auto costheta = dot(wh, n);

		if (costheta <= 0) {
			return 0;
		}

		auto cos2 = costheta * costheta;

		auto sintheta = aten::sqrt(1 - aten::clamp<real>(cos2, 0, 1));
		auto tantheta = sintheta / costheta;
		auto tan2 = tantheta * tantheta;

		real a = roughness;
		auto a2 = a * a;

		auto D = 1 / (AT_MATH_PI * a2 * cos2 * cos2);
		D *= aten::exp(-tan2 / a2);

		return D;
	}

	real MicrofacetBeckman::pdf(
		real roughness,
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
	{
		// NOTE
		// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

		auto wh = normalize(-wi + wo);

		auto costheta = aten::abs(dot(wh, normal));

		auto D = sampleBeckman_D(wh, normal, roughness);

		auto denom = 4 * aten::abs(dot(wo, wh));

		auto pdf = denom > 0 ? (D * costheta) / denom : 0;

		return pdf;
	}

	vec3 MicrofacetBeckman::sampleDirection(
		real roughness,
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		// NOTE
		// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-bsdf/

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		auto a = roughness;
		auto a2 = a * a;

		auto theta = aten::sqrt(-a2 * aten::log(1 - r1 * 0.99));
		theta = aten::atan(theta);
		theta = ((theta >= 0) ? theta : (theta + 2 * AT_MATH_PI));

		auto phi = 2 * AT_MATH_PI * r2;

		auto costheta = aten::cos(theta);
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		auto cosphi = aten::cos(phi);
		auto sinphi = aten::sqrt(1 - cosphi * cosphi);

		// Ortho normal base.
		auto n = normal;
		auto t = getOrthoVector(normal);
		auto b = normalize(cross(n, t));

		auto w = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
		w.normalize();

		auto dir = in - 2 * dot(in, w) * w;

		return std::move(dir);
	}

	vec3 MicrofacetBeckman::bsdf(
		real roughness,
		real& fresnel,
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		// ÉåÉCÇ™ì¸éÀÇµÇƒÇ≠ÇÈë§ÇÃï®ëÃÇÃã¸ê‹ó¶.
		real ni = real(1);	// ê^ãÛ

		real nt = ior();	// ï®ëÃì‡ïîÇÃã¸ê‹ó¶.

		vec3 V = -wi;
		vec3 L = wo;
		vec3 N = normal;
		vec3 H = normalize(L + V);

		// TODO
		// DesneyÇæÇ∆absÇµÇƒÇ»Ç¢Ç™ÅAAMDÇÃÇÕÇµÇƒÇ¢ÇÈ....
		auto NdotH = aten::abs(dot(N, H));
		auto VdotH = aten::abs(dot(V, H));
		auto NdotL = aten::abs(dot(N, L));
		auto NdotV = aten::abs(dot(N, V));

		auto a = roughness;

		// Compute D.
		real D = sampleBeckman_D(H, N, a);

		// Compute G.
		real G(1);
		{
			// NOTE
			// http://graphicrants.blogspot.jp/2013/08/specular-bsdf-reference.html

			auto c = NdotV < 1 ? NdotV / (a * aten::sqrt(1 - NdotV * NdotV)) : 0;
			auto c2 = c * c;

			if (c < 1.6) {
				G = (3.535 * c + 2.181 * c2) / (1 + 2.276 * c + 2.577 * c2);
			}
			else {
				G = 1;
			}
		}

		auto albedo = color();
		albedo *= sampleAlbedoMap(u, v);

		real F(1);
		{
			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) Å‡ R0 + (1 - R0)(1 - cosÉ¶)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (ni - nt) / (ni + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(L, H));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		auto denom = 4 * NdotL * NdotV;

		auto bsdf = denom > AT_MATH_EPSILON ? albedo * F * G * D / denom : 0;

		fresnel = F;

		return std::move(bsdf);
	}

	material::sampling MicrofacetBeckman::sample(
		const vec3& in,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		auto roughness = sampleRoughness(u, v);

		ret.dir = sampleDirection(roughness, in, normal, sampler);
		ret.pdf = pdf(roughness, normal, in, ret.dir);

		real fresnel = 1;
		ret.bsdf = bsdf(roughness, fresnel, normal, in, ret.dir, u, v);
		ret.fresnel = fresnel;

		return std::move(ret);
	}
}
