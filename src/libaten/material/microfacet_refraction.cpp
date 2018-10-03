#include "material/microfacet_refraction.h"
#include "material/ggx.h"

#pragma optimize( "", off)

namespace AT_NAME
{
	AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		auto ret = pdf(roughness.r, param->ior, normal, wi, wo);
		return ret;
	}

	AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return pdf(&m_param, normal, wi, wo, u, v);
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		aten::vec3 dir = sampleDirection(roughness.r, param->ior, normal, wi, sampler);

		return std::move(dir);
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		auto albedo = param->baseColor;
		albedo *= material::sampleTexture(param->albedoMap, u, v, real(1));

		real fresnel = 1;
		real ior = param->ior;

		aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
		return std::move(ret);
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v,
		const aten::vec3& externalAlbedo)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		auto albedo = param->baseColor;
		albedo *= externalAlbedo;

		real fresnel = 1;
		real ior = param->ior;

		aten::vec3 ret = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, wo, u, v);
		return std::move(ret);
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(&m_param, normal, wi, wo, u, v));
	}

	AT_DEVICE_MTRL_API real MicrofacetRefraction::pdf(
		real roughness,
		real ior,
		const aten::vec3& nml,
		const aten::vec3& wi,
		const aten::vec3& wo)
	{
		const auto& i = -wi;
		const auto& o = wo;

		bool isSameHemisphere = dot(i, o) > real(0);
		if (isSameHemisphere) {
			return real(0);
		}

		auto pdf = MicrofacetGGX::pdf(roughness, nml, wi, wo);


		bool into = dot(i, nml) > real(0);

		real nc = real(1);	// 真空の屈折率.
		real nt = ior;		// 物体内部の屈折率.
		real nnt = into ? nc / nt : nt / nc;

		// Macrofacet normal.
		const auto& n = into ? nml : -nml;

		//const auto wh = normalize(i + nnt * o);
		auto wh = i + nnt * o;
		{
			auto len = aten::length(wh);
			wh = len > real(0) ? wh / len : wh;
		}

		const float denom = dot(o, wh) + nnt * dot(i, wh);
		const float dwh_dwi = nnt * nnt * aten::abs(dot(i, wh)) / (denom * denom + real(0.000001));

		pdf = pdf * dwh_dwi;

		return pdf == real(0) ? real(1) : pdf;
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::sampleDirection(
		real roughness,
		real ior,
		const aten::vec3& wi,
		const aten::vec3& nml,
		aten::sampler* sampler)
	{
#if 0
		const auto& i = -wi;

		bool into = dot(i, nml) > real(0);

		// Macrofacet normal.
		const auto& n = into ? nml : -nml;

		real nc = real(1);	// 真空の屈折率.
		real nt = ior;		// 物体内部の屈折率.
		real nnt = into ? nc / nt : nt / nc;
		real ddn = dot(wi, nml);

		// TODO
		// 全反射.

		// Sample Microfacet normal.
		const auto m = MicrofacetGGX::sampleNormal(roughness, n, sampler);

#if 0
		const real sign = into ? real(1) : real(-1);
		const real c = dot(i, m);
		
		//const auto dir = (nnt * c - sign * aten::sqrt(real(1) + nnt * (c * c - real(1)))) * m - nnt * i;

		const auto t = real(1) - nnt * nnt * aten::cmpMax(real(0), real(1) - c * c);

		if (t < real(0)) {
			return aten::vec3(0);
		}

		const auto tt = nnt * c - sign * aten::sqrt(t);

		auto dir = tt * m;
		dir = dir - nnt * i;

		auto xxx = dot(i, dir);
#else
		const real coso = dot(i, m);
		const real eta = coso > 0 ? (nt / nc) : (nc / nt);
		const real t = real(1) - eta * eta * std::max(0.0f, 1.0f - coso * coso);

		// total inner reflection
		if (t <= 0.0f) {
			auto reflect = wi - 2 * dot(nml, wi) * nml;
			reflect = normalize(reflect);
			return aten::vec3(0);
		}
		const float scale = coso < 0.0f ? -1.0f : 1.0f;
		auto dir = -eta * i + (eta * coso - scale * sqrt(t)) * n;
#endif
#else
		aten::vec3 in = -wi;
		aten::vec3 n = nml;

		bool into = (dot(in, n) > real(0));

		if (!into) {
			n = -n;
		}

		real nc = real(1);		// 真空の屈折率.
		real nt = ior;	// 物体内部の屈折率.
		real nnt = into ? nc / nt : nt / nc;
		real ddn = dot(wi, nml);

		// NOTE
		// cos_t^2 = 1 - sin_t^2
		// sin_t^2 = (nc/nt)^2 * sin_i^2
		//         = (nc/nt)^2 * (1 - cos_i^2)
		// sin_i / sin_t = nt/nc
		//   -> sin_t = (nc/nt) * sin_i
		//            = (nc/nt) * sqrt(1 - cos_i)
		real cos2t = real(1) - nnt * nnt * (real(1) - ddn * ddn);

		const auto m = MicrofacetGGX::sampleNormal(roughness, n, sampler);

		if (cos2t < real(0)) {
			auto reflect = wi - 2 * dot(m, wi) * m;
			reflect = normalize(reflect);
			return std::move(reflect);
		}

		auto d = dot(in, m);
		auto dir = -nnt * (in - d * m) - aten::sqrt(real(1) - nnt * nnt * (1 - ddn * ddn)) * m;
		dir = normalize(dir);

		auto xxx = dot(in, dir);
#endif

		return std::move(dir);
	}

	AT_DEVICE_MTRL_API aten::vec3 MicrofacetRefraction::bsdf(
		const aten::vec3& albedo,
		const real roughness,
		const real ior,
		real& fresnel,
		const aten::vec3& nml,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		const aten::vec3& i = -wi;
		const aten::vec3& o = wo;

		bool into = dot(i, nml) > real(0);

		const aten::vec3& n = into ? nml : -nml;

		real nc = real(1);	// 真空
		real nt = ior;		// 物体内部の屈折率.
		real nnt = into ? nc / nt : nt / nc;

		//const auto wh = normalize(i + nnt * o);
		auto wh = i + nnt * o;
		{
			auto len = aten::length(wh);
			wh = len > real(0) ? wh / len : wh;
		}

		real D = MicrofacetGGX::sampleGGX_D(wh, n, roughness);

		// Compute G.
		real G(1);
		{
			auto G1_lh = MicrofacetGGX::computeGGXSmithG1(roughness, i, n);
			auto G1_vh = MicrofacetGGX::computeGGXSmithG1(roughness, o, n);

			G = G1_lh * G1_vh;
		}

		real F(1);
		{
			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (nc - nt) / (nc + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(o, wh));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		auto IdotN = aten::abs(dot(i, n));
		auto OdotN = aten::abs(dot(o, n));

		auto IdotH = aten::abs(dot(i, wh));
		auto OdotH = aten::abs(dot(o, wh));

		auto NdotV = aten::abs(dot(n, i));

		//auto denom = IdotH + nnt * OdotH;

		const float sqrtDenom = IdotH + nnt * OdotH;
		const float t = nnt / (sqrtDenom + 0.0001);

#if 0
		//auto bsdf = denom > AT_MATH_EPSILON ? albedo * nnt * nnt * (real(1) - F) * G * D * ((IdotH * OdotH) / (IdotN * OdotN)) / (denom * denom) : aten::vec3(0);
		auto bsdf = denom > AT_MATH_EPSILON
			? albedo * ((real(1) - F) * aten::abs(G * D * (nnt * denom) * (nnt * denom) * IdotH * OdotH / denom))
			: aten::vec3(0);
#else
		auto bsdf = albedo * ((real(1) - F) * aten::abs(G * D * t * t * IdotH * OdotH / NdotV));
#endif

		fresnel = F;

		return std::move(bsdf);
	}

	AT_DEVICE_MTRL_API void MicrofacetRefraction::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		result->dir = sampleDirection(roughness.r, param->ior, wi, normal, sampler);

		auto l = aten::length(result->dir);
		if (l == real(0)) {
			result->pdf = real(0);
			return;
		}

		result->pdf = pdf(roughness.r, param->ior, normal, wi, result->dir);

		auto albedo = param->baseColor;
		albedo *= material::sampleTexture(param->albedoMap, u, v, real(1));

		real fresnel = 1;
		real ior = param->ior;

		result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
		result->fresnel = fresnel;
	}

	AT_DEVICE_MTRL_API void MicrofacetRefraction::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		const aten::vec3& externalAlbedo,
		bool isLightPath/*= false*/)
	{
		auto roughness = material::sampleTexture(
			param->roughnessMap,
			u, v,
			param->roughness);

		result->dir = sampleDirection(roughness.r, param->ior, wi, normal, sampler);
		result->pdf = pdf(roughness.r, param->ior, normal, wi, result->dir);

		auto albedo = param->baseColor;
		albedo *= externalAlbedo;

		real fresnel = 1;
		real ior = param->ior;

		result->bsdf = bsdf(albedo, roughness.r, ior, fresnel, normal, wi, result->dir, u, v);
		result->fresnel = fresnel;
	}

	AT_DEVICE_MTRL_API MaterialSampling MicrofacetRefraction::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		MaterialSampling ret;

		sample(
			&ret,
			&m_param,
			normal,
			ray.dir,
			orgnormal,
			sampler,
			u, v,
			isLightPath);

		return std::move(ret);
	}

	bool MicrofacetRefraction::edit(aten::IMaterialParamEditor* editor)
	{
		auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param, roughness);
		auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
		auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

		AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
		AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
		AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

		return b0 || b1 || b2;
	}
}
