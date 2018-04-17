#include "math/math.h"
#include "material/disney_brdf.h"
#include "material/lambert.h"

namespace AT_NAME
{
	// NOTE
	// http://project-asura.com/blog/archives/1972
	// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
	// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf

	inline AT_DEVICE_API real sqr(real f)
	{
		return f * f;
	}

	inline AT_DEVICE_API real SchlickFresnel(real u)
	{
		real m = aten::clamp<real>(1 - u, 0, 1);
		real m2 = m * m;
		return m2 * m2 * m; // pow(m,5)
	}

	inline AT_DEVICE_API real SchlickFresnelEta(real eta, real cosi)
	{
		const real f = ((real(1) - eta) / (real(1) + eta)) * ((real(1) - eta) / (real(1) + eta));
		const real m = real(1) - aten::abs(cosi);
		const real m2 = m * m;
		return f + (real(1) - f) * m2 * m2 * m;
	}

	inline AT_DEVICE_API real GTR1(real NdotH, real a)
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

	inline AT_DEVICE_API real GTR2(real NdotH, real a)
	{
		// NOTE
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p25 equation(8)

		real a2 = a * a ;
		real t = 1 + (a2 - 1) * NdotH * NdotH;
		return a2 / (AT_MATH_PI * t * t);
	}

	inline AT_DEVICE_API real GTR2_aniso(real NdotH, real HdotX, real HdotY, real ax, real ay)
	{
		// NOTE
		// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		// p25 equation(13)

#if 0
		return 1 / (AT_MATH_PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
#else
		// A trick to avoid fireflies from RadeonRaySDK.
		auto f = sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH);
		f = AT_MATH_PI * ax * ay * f * f;
		
		f = aten::clamp<real>(1 / f, 0, 10);

		return f;
#endif
	}

	inline AT_DEVICE_API real smithG_GGX(real NdotV, real alphaG)
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

	inline AT_DEVICE_API real smithG_GGX_aniso(real NdotV, real VdotX, real VdotY, real ax, real ay)
	{
		return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
	}

	inline AT_DEVICE_API aten::vec3 mon2lin(const aten::vec3& x)
	{
		return aten::vec3(aten::pow(x[0], real(2.2)), aten::pow(x[1], real(2.2)), aten::pow(x[2], real(2.2)));
	}

	AT_DEVICE_MTRL_API real DisneyBRDF::pdf(
		const aten::vec3& normal,
		const aten::vec3& wi,	/* in */
		const aten::vec3& wo,	/* out */
		real u, real v) const
	{
		return pdf(&m_param, normal, wi, wo, u, v);
	}

	AT_DEVICE_MTRL_API real DisneyBRDF::pdf(
		const aten::MaterialParameter* mtrl,
		const aten::vec3& normal,
		const aten::vec3& wi,	/* in */
		const aten::vec3& wo,	/* out */
		real u, real v)
	{
		const aten::vec3& N = normal;
		const aten::vec3& V = -wi;
		const aten::vec3& L = wo;
		
		const aten::vec3 X = aten::getOrthoVector(N);
		const aten::vec3 Y = normalize(cross(N, X));
		
		// TODO
		const auto anisotropic = mtrl->anisotropic;
		const auto roughness = mtrl->roughness;
		const auto clearcoatGloss = mtrl->clearcoatGloss;
		const auto clearcoat = mtrl->clearcoat;
		const auto baseColor = mtrl->baseColor;
		const auto specular = mtrl->specular;
		const auto specular_tint = mtrl->specularTint;
		const auto metallic = mtrl->metallic;

		const auto ax = aten::cmpMax<real>(real(0.001), roughness * roughness * (1 + anisotropic));	// roughness for x direction.
		const auto ay = aten::cmpMax<real>(real(0.001), roughness * roughness * (1 - anisotropic));	// roughness for y direction.

		const auto H = normalize(L + V);

		const auto NdotH = dot(N, H);
		const auto HdotX = dot(H, X);
		const auto HdotY = dot(H, Y);
		const auto Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay);

		const auto LdotH = dot(L, H);

		const auto specularPdf = (Ds * NdotH) / (real(4) * LdotH);

		const auto diffusePdf = aten::abs(dot(L, N)) / AT_MATH_PI;

		const auto gtr1 = GTR1(NdotH, aten::mix(real(0.1), real(0.001), clearcoatGloss));
		const auto clearcoatPdf = (gtr1 * NdotH) / (real(4) * LdotH);

		aten::vec3 cd_lin = aten::pow(baseColor, real(1 / 2.2));

		// Luminance approximmation
		real cd_lum = dot(cd_lin, aten::vec3(0.3f, 0.6f, 0.1f));

		// Normalize lum. to isolate hue+sat
		aten::vec3 c_tint = cd_lum > 0.f ? (cd_lin / cd_lum) : aten::vec3(1);

		aten::vec3 c_spec0 = mix(
			specular * real(0.1) * mix(aten::vec3(1), c_tint, specular_tint),
			cd_lin, 
			metallic);

		real cs_lum = dot(c_spec0, aten::vec3(0.3f, 0.6f, 0.1f));

		real cs_w = cs_lum / (cs_lum + (1.f - metallic) * cd_lum);

		auto ret = clearcoatPdf * clearcoat + (1.f - clearcoat) * (cs_w * specularPdf + (1.f - cs_w) * diffusePdf);

		return ret;
	}

	AT_DEVICE_MTRL_API aten::vec3 DisneyBRDF::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
	}

	AT_DEVICE_MTRL_API aten::vec3 DisneyBRDF::sampleDirection(
		const aten::MaterialParameter* mtrl,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		const auto roughness = mtrl->roughness;
		const auto anisotropic = mtrl->anisotropic;
		const auto clearcoat = mtrl->clearcoat;
		const auto clearcoatGloss = mtrl->clearcoatGloss;
		const auto baseColor = mtrl->baseColor;
		const auto specular = mtrl->specular;
		const auto specular_tint = mtrl->specularTint;
		const auto metallic = mtrl->metallic;

		const auto ax = aten::cmpMax(real(0.001), roughness * roughness * (real(1) + anisotropic));
		const auto ay = aten::cmpMax(real(0.001), roughness * roughness * (real(1) - anisotropic));

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		aten::vec3 dir;

		if (r1 < clearcoat) {
			// Clearcoat.

			// Normalize [0, 1].
			r1 /= clearcoat;

			const auto a = aten::mix(real(0.1), real(0.001), clearcoatGloss);

			// NOTE
			// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
			// p25 (4) (5)

			const auto costheta = aten::sqrt(real(1) - aten::pow(a * a, real(1) - r2) / (real(1) - a * a));
			const auto sintheta = aten::sqrt(real(1) - costheta * costheta);

			const auto phi = real(2) * AT_MATH_PI * r1;
			const auto cosphi = aten::cos(phi);
			const auto sinphi = aten::sin(phi);

			auto n = normal;
			auto t = aten::getOrthoVector(n);
			auto b = cross(n, t);

			//aten::vec3 wh = aten::vec3(sintheta * cosphi, sintheta * sinphi, costheta);
			aten::vec3 wh = t * sintheta * cosphi + b * sintheta * sinphi + n * costheta;
			wh = normalize(wh);

			dir = wi - 2 * dot(wi, wh) * wh;
		}
		else {
			// Reduce clear coat ant normalize.
			r1 -= clearcoat;
			r1 /= (real(1) - clearcoat);

			aten::vec3 cd_lin = aten::pow(baseColor, real(1 / 2.2));

			// Luminance approximmation
			real cd_lum = dot(cd_lin, aten::vec3(0.3f, 0.6f, 0.1f));

			// Normalize lum. to isolate hue+sat
			aten::vec3 c_tint = cd_lum > 0.f ? (cd_lin / cd_lum) : aten::vec3(1);

			aten::vec3 c_spec0 = mix(
				specular * real(0.1) * mix(aten::vec3(1), c_tint, specular_tint),
				cd_lin,
				metallic);

			real cs_lum = dot(c_spec0, aten::vec3(0.3f, 0.6f, 0.1f));

			real cs_w = cs_lum / (cs_lum + (1.f - metallic) * cd_lum);

			if (r2 < cs_w) {
				// Specular.

				auto n = normal;
				auto t = aten::getOrthoVector(n);
				auto b = cross(n, t);

				// NOTE
				// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
				// p25 (14)

				const auto f = aten::sqrt(r2 / (real(1) - r2));

#if 0
				aten::vec3 wh = f * aten::vec3(
					ax * aten::cos(real(2) * AT_MATH_PI * r1),
					ay * aten::sin(real(2) * AT_MATH_PI * r1),
					real(1));
#else
				aten::vec3 wh = f * (t * ax * aten::cos(real(2) * AT_MATH_PI * r1)
					+ b * ay * aten::sin(real(2) * AT_MATH_PI * r1)
					+ n);
#endif
				wh = normalize(wh);

				dir = wi - 2 * dot(wi, wh) * wh;
			}
			else {
				// Diffuse.

				// Reduce diffse and normalize.
				r2 -= cs_w;
				r2 /= (real(1) - cs_w);

				auto n = normal;
				auto t = aten::getOrthoVector(n);
				auto b = cross(n, t);

				// コサイン項を使った重点的サンプリング.
				r1 = 2 * AT_MATH_PI * r1;
				auto r2s = aten::sqrt(r2);

				const real x = aten::cos(r1) * r2s;
				const real y = aten::sin(r1) * r2s;
				const real z = aten::sqrt(real(1) - r2);

				dir = normalize((t * x + b * y + n * z));
			}
		}

		return std::move(dir);
	}

	AT_DEVICE_MTRL_API aten::vec3 DisneyBRDF::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(&m_param, normal, wi, wo, u, v));
	}

	AT_DEVICE_MTRL_API aten::vec3 DisneyBRDF::bsdf(
		const aten::MaterialParameter* mtrl,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		const aten::vec3& N = normal;
		const aten::vec3 V = -wi;
		const aten::vec3& L = wo;

		const aten::vec3 X = aten::getOrthoVector(N);
		const aten::vec3 Y = normalize(cross(N, X));

		const auto baseColor = mtrl->baseColor;
		const auto subsurface = mtrl->subsurface;
		const auto metalic = mtrl->metallic;
		const auto specular = mtrl->specular;
		const auto specularTint = mtrl->specularTint;
		const auto roughness = mtrl->roughness;
		const auto anisotropic = mtrl->anisotropic;
		const auto sheen = mtrl->sheen;
		const auto sheenTint = mtrl->sheenTint;
		const auto clearcoat = mtrl->clearcoat;
		const auto clearcoatGloss = mtrl->clearcoatGloss;

		auto NdotL = dot(N, L);
		auto NdotV = dot(N, V);

#if 0
		if (NdotL < 0 || NdotV < 0) {
			return aten::vec3(0);
		}
#else
		NdotL = aten::abs(NdotL);
		NdotV = aten::abs(NdotV);
#endif

		const aten::vec3 H = normalize(L + V);
		const auto NdotH = dot(N, H);
		const auto LdotH = dot(L, H);

		const aten::vec3 Cdlin = mon2lin(baseColor);
		const auto Cdlum = real(0.3) * Cdlin[0] + real(0.6) * Cdlin[1] + real(0.1) * Cdlin[2]; // luminance approx.

		const aten::vec3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : aten::vec3(real(1)); // normalize lum. to isolate hue+sat
		const aten::vec3 Cspec0 = glm::mix(specular* real(0.08) * glm::mix(aten::vec3(real(1)), Ctint, specularTint), Cdlin, metalic);
		const aten::vec3 Csheen = glm::mix(aten::vec3(1), Ctint, sheenTint);

		// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
		// and mix in diffuse retro-reflection based on roughness
		const auto FL = SchlickFresnel(NdotL);
		const auto FV = SchlickFresnel(NdotV);
		const auto Fd90 = real(0.5) + real(2) * LdotH * LdotH * roughness;
		const auto Fd = aten::mix(real(1), Fd90, FL) * aten::mix(real(1), Fd90, FV);

		// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
		// 1.25 scale is used to (roughly) preserve albedo
		// Fss90 used to "flatten" retroreflection based on roughness
		const auto Fss90 = LdotH * LdotH * roughness;
		const auto Fss = aten::mix(real(1), Fss90, FL) * aten::mix(real(1), Fss90, FV);
		const auto ss = real(1.25) * (Fss * (real(1) / (NdotL + NdotV) - real(0.5)) + real(0.5));

		// specular
		const auto aspect = aten::sqrt(1 - anisotropic * real(0.9));
		const auto ax = aten::cmpMax(real(0.001), sqr(roughness) / aspect);
		const auto ay = aten::cmpMax(real(0.001), sqr(roughness) * aspect);
		const auto Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
		const auto FH = SchlickFresnel(LdotH);
		const aten::vec3 Fs = aten::mix(Cspec0, aten::vec3(real(1)), FH);
		real Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
		Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

		// sheen
		aten::vec3 Fsheen = FH * sheen * Csheen;

		// clearcoat (ior = 1.5 -> F0 = 0.04)
		const auto Dr = GTR1(NdotH, aten::mix(real(0.1), real(0.001), clearcoatGloss));
		const auto Fr = aten::mix(real(0.04), real(real(1)), FH);
		const auto Gr = smithG_GGX(NdotL, real(0.25)) * smithG_GGX(NdotV, real(0.25));	// 論文内で0.25決めうちと記載.

		auto ret = ((1 / AT_MATH_PI) * aten::mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (1 - metalic)	// diffuse
			+ Gs * Fs * Ds						// specular
#if 1
			+ clearcoat * Gr * Fr * Dr;	// clearcoat
#else
			// A trick to avoid fireflies from RadeonRaySDK.
			+ aten::clamp<real>(clearcoat * Gr * Fr * Dr, real(0), real(0.5));	// clearcoat.
#endif

		return ret;
	}

	AT_DEVICE_MTRL_API MaterialSampling DisneyBRDF::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		MaterialSampling ret;
		sample(&ret, &m_param, normal, ray.dir, orgnormal, sampler, u, v, isLightPath);

		return std::move(ret);
	}

	AT_DEVICE_MTRL_API void DisneyBRDF::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* mtrl,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath)
	{
		result->dir = sampleDirection(mtrl, normal, wi, u, v, sampler);

		const auto wo = result->dir;

		result->pdf = pdf(mtrl, normal, wi, wo, u, v);
		result->bsdf = bsdf(mtrl, normal, wi, wo, u, v);
	}

	AT_DEVICE_MTRL_API real DisneyBRDF::computeFresnel(
		const aten::MaterialParameter* mtrl,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real outsideIor)
	{
		real fresnel = real(1);

		const real et = 1.f;
		if (et != 0.f)
		{
			const real cosi = dot(normal, -wi);
			fresnel = real(1) - SchlickFresnelEta(et, cosi);
		}

		return fresnel;
	}

	bool DisneyBRDF::edit(aten::IMaterialParamEditor* editor)
	{
		bool b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, subsurface, 0, 1);
		bool b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, metallic, 0, 1);
		bool b2 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, specular, 0, 1);
		bool b3 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, specularTint, 0, 1);
		bool b4 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, roughness, 0, 1);
		bool b5 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, anisotropic, 0, 1);
		bool b6 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, sheen, 0, 1);
		bool b7 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, sheenTint, 0, 1);
		bool b8 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, clearcoat, 0, 1);
		bool b9 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, clearcoatGloss, 0, 1);
		bool b10 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
		bool b11 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

		AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
		AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

		return b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 || b9 || b10 || b11;
	}
}
