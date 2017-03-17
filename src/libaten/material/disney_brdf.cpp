#include "math/math.h"
#include "material/disney_brdf.h"

namespace aten
{
	// NOTE
	// http://project-asura.com/blog/archives/1972
	// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
	// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf

	inline real sqr(real f)
	{
		return f * f;
	}

	inline real SchlickFresnel(real u)
	{
		real m = aten::clamp<real>(1 - u, 0, 1);
		real m2 = m * m;
		return m2 * m2 * m; // pow(m,5)
	}

	inline real GTR1(real NdotH, real a)
	{
		// NOTE
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p25 equation(4)

		if (a >= 1) {
			return 1 / AT_MATH_PI;
		}
		real a2 = a * a;
		real t = 1 + (a2 - 1) * NdotH * NdotH;
		return (a2 - 1) / (AT_MATH_PI * aten::log(a2) * t);
	}

	inline real GTR2(real NdotH, real a)
	{
		// NOTE
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p25 equation(8)

		real a2 = a *a ;
		real t = 1 + (a2 - 1) * NdotH * NdotH;
		return a2 / (AT_MATH_PI * t*t);
	}

	inline real GTR2_aniso(real NdotH, real HdotX, real HdotY, real ax, real ay)
	{
		// NOTE
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p25 equation(13)

#if 0
		return 1 / (AT_MATH_PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
#else
		// A trick to avoid fireflies from RadeonRaySDK.
		auto f = 1 / (AT_MATH_PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
		f = aten::clamp<real>(f, 0, 10);
		return f;
#endif
	}

	inline real smithG_GGX(real NdotV, real alphaG)
	{
		// NOTE
		// http://graphicrants.blogspot.jp/2013/08/specular-brdf-reference.html
		// Geometric Shadowing - Smith - GGX

		real a = alphaG * alphaG;
		real b = NdotV * NdotV;
#if 1
		// Disney BRDF.
		return 1 / (NdotV + sqrt(a + b - a * b));
#else
		// Smith GGX
		return (2 * NdotV) / (NdotV + sqrt(a + b - a * b));
#endif
	}

	inline real smithG_GGX_aniso(real NdotV, real VdotX, real VdotY, real ax, real ay)
	{
		return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
	}

	inline vec3 mon2lin(const vec3& x)
	{
		return vec3(aten::pow(x[0], 2.2), aten::pow(x[1], 2.2), aten::pow(x[2], 2.2));
	}

	real DisneyBRDF::pdf(
		const vec3& normal,
		const vec3& wi,	/* in */
		const vec3& wo,	/* out */
		real u, real v,
		sampler* sampler) const
	{
		const vec3& N = normal;
		const vec3& V = -wi;
		const vec3& L = wo;
		
		const vec3 X = getOrthoVector(N);
		const vec3 Y = normalize(cross(N, X));

		return pdf(V, N, L, X, Y, u, v);
	}

	real DisneyBRDF::pdf(
		const vec3& V,
		const vec3& N,
		const vec3& L,
		const vec3& X,
		const vec3& Y,
		real u, real v) const
	{
		const auto VdotN = dot(V, N);
		const auto LdotN = dot(L, N);

		if (VdotN < 0 || LdotN < 0) {
			return 0;
		}

		const auto anisotropic = m_anisotropic;
		const auto roughness = m_roughness;
		const auto metalic = m_metallic;

		const auto weight2 = metalic;
		const auto weight1 = 1 - weight2;

		// diffuse.
		const real diffusePdf = LdotN / AT_MATH_PI;

		// specular
		const auto aspect = aten::sqrt(1 - anisotropic * 0.9);
		const auto ax = std::max<real>(0.001, sqr(roughness) / aspect);	// roughness for x direction.
		const auto ay = std::max<real>(0.001, sqr(roughness) * aspect);	// roughness for y direction.

		vec3 H = normalize(V + L);

		const auto NdotH = dot(N, H);
		const auto HdotX = dot(H, X);
		const auto HdotY = dot(H, Y);
		const auto LdotH = dot(L, H);

		const auto Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay);
		const auto costhetah = NdotH;

		// NOTE
		// pdfl = pdfh / (4 * dot(l, h))
		// pdfh = D(θh) * cos(θh)
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p24
		const real specularPdf = (Ds * costhetah) / (4 * LdotH);

		real pdf = weight1 * diffusePdf + weight2 * specularPdf;

		return pdf;
	}

	vec3 DisneyBRDF::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		const vec3& in = ray.dir;

		const vec3& N = normal;
		const vec3& V = -in;

		const vec3 X = getOrthoVector(N);
		const vec3 Y = normalize(cross(N, X));

		return std::move(sampleDirection(V, N, X, Y, u, v, sampler));
	}

	vec3 DisneyBRDF::sampleDirection(
		const vec3& V,
		const vec3& N,
		const vec3& X,
		const vec3& Y,
		real u, real v,
		sampler* sampler) const
	{
		const auto anisotropic = m_anisotropic;
		const auto roughness = m_roughness;
		const auto metalic = m_metallic;

		const auto weight2 = metalic;
		const auto weight1 = 1 - weight2;

		// specular
		const auto aspect = aten::sqrt(1 - anisotropic * 0.9);
		const auto ax = std::max<real>(0.001, sqr(roughness) / aspect);	// roughness for x direction.
		const auto ay = std::max<real>(0.001, sqr(roughness) * aspect);	// roughness for y direction.

		const auto r = sampler->nextSample();

		bool willSampleDiffuse = r < weight1;

		vec3 H;
		vec3 L;

		if (willSampleDiffuse) {
			// Sample diffuse.

			// コサイン項を使った重点的サンプリング.
			const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
			const real r2 = sampler->nextSample();
			const real r2s = sqrt(r2);

			const real x = aten::cos(r1) * r2s;
			const real y = aten::sin(r1) * r2s;
			const real z = aten::sqrt(real(1) - r2);

			vec3 n = N;
			vec3 t = X;
			vec3 b = Y;

			L = normalize((t * x + b * y + n * z));
			H = normalize(L + V);
		}
		else {
			// Sample specular.
			auto r1 = sampler->nextSample();
			auto r2 = sampler->nextSample();

			// NOTE
			// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
			// p24 - p26 equation(14)
			H = sqrt(r2 / (1 - r2)) * (ax * aten::cos(2 * AT_MATH_PI * r1) + ay * sin(2 * AT_MATH_PI * r1)) + N;
			H.normalize();

			L = 2 * dot(V, H) * H - V;
		}

		return std::move(L);
	}

	vec3 DisneyBRDF::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		const vec3& N = normal;
		const vec3 V = -wi;
		const vec3& L = wo;

		const vec3 X = getOrthoVector(N);
		const vec3 Y = normalize(cross(N, X));

		real fresnel = 1;

		return std::move(bsdf(fresnel, V, N, L, X, Y, u, v));
	}

	vec3 DisneyBRDF::bsdf(
		real& fresnel,
		const vec3& V,
		const vec3& N,
		const vec3& L,
		const vec3& X,
		const vec3& Y,
		real u, real v) const
	{
		const auto baseColor = m_baseColor;
		const auto subsurface = m_subsurface;
		const auto metalic = m_metallic;
		const auto specular = m_specular;
		const auto specularTint = m_specularTint;
		const auto roughness = m_roughness;
		const auto anisotropic = m_anisotropic;
		const auto sheen = m_sheen;
		const auto sheenTint = m_sheenTint;
		const auto clearcoat = m_clearcoat;
		const auto clearcoatGloss = m_clearcoatGloss;

		const auto NdotL = dot(N, L);
		const auto NdotV = dot(N, V);
		if (NdotL < 0 || NdotV < 0) {
			return vec3(0);
		}

		const vec3 H = normalize(L + V);
		const auto NdotH = dot(N, H);
		const auto LdotH = dot(L, H);

		const vec3 Cdlin = mon2lin(baseColor);
		const auto Cdlum = 0.3 * Cdlin[0] + 0.6 * Cdlin[1] + 0.1 * Cdlin[2]; // luminance approx.

		const vec3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : vec3(1); // normalize lum. to isolate hue+sat
		const vec3 Cspec0 = mix(specular* 0.08 * mix(vec3(1), Ctint, specularTint), Cdlin, metalic);
		const vec3 Csheen = mix(vec3(1), Ctint, sheenTint);

		// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
		// and mix in diffuse retro-reflection based on roughness
		const auto FL = SchlickFresnel(NdotL);
		const auto FV = SchlickFresnel(NdotV);
		const auto Fd90 = 0.5 + 2 * LdotH * LdotH * roughness;
		const auto Fd = aten::mix(real(1), Fd90, FL) * aten::mix(real(1), Fd90, FV);

		// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
		// 1.25 scale is used to (roughly) preserve albedo
		// Fss90 used to "flatten" retroreflection based on roughness
		const auto Fss90 = LdotH * LdotH * roughness;
		const auto Fss = aten::mix(real(1), Fss90, FL) * aten::mix(real(1), Fss90, FV);
		const auto ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

		// specular
		const auto aspect = aten::sqrt(1 - anisotropic * 0.9);
		const auto ax = std::max(0.001, sqr(roughness) / aspect);
		const auto ay = std::max(0.001, sqr(roughness) * aspect);
		const auto Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
		const auto FH = SchlickFresnel(LdotH);
		const vec3 Fs = aten::mix(Cspec0, vec3(1), FH);
		real Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
		Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

		// sheen
		vec3 Fsheen = FH * sheen * Csheen;

		// clearcoat (ior = 1.5 -> F0 = 0.04)
		const auto Dr = GTR1(NdotH, aten::mix(0.1, 0.001, clearcoatGloss));
		const auto Fr = aten::mix(0.04, real(1), FH);
		const auto Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);	// 論文内で0.25決めうちと記載.

		// TODO
		{
			const auto weight2 = metalic;
			const auto weight1 = 1 - weight2;

			fresnel = weight1 * aten::mix(Fd, ss, subsurface) + weight2 * FH;
		}

		return ((1 / AT_MATH_PI) * aten::mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (1 - metalic)	// diffuse
			+ Gs * Fs * Ds						// specular
#if 0
			+ 0.25 * clearcoat * Gr * Fr * Dr;	// clearcoat
#else
			// A trick to avoid fireflies from RadeonRaySDK.
			+ aten::clamp<real>(clearcoat * Gr * Fr * Dr, 0, 0.5);	// clearcoat.
#endif
	}

	material::sampling DisneyBRDF::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		const vec3& in = ray.dir;

		const vec3& N = normal;
		const vec3 V = -in;

		const vec3 X = getOrthoVector(normal);
		const vec3 Y = normalize(cross(normal, X));

		sampling ret;

		ret.dir = sampleDirection(V, N, X, Y, u, v, sampler);

		const vec3& L = ret.dir;

		ret.pdf = pdf(V, N, L, X, Y, u, v);

		real fresnel = 1;
		ret.bsdf = bsdf(fresnel, V, N, L, X, Y, u, v);
		ret.fresnel = fresnel;

		return std::move(ret);
	}
}
