#include "material/carpaint.h"
#include "material/ggx.h"
#include "material/FlakesNormal.h"

//#pragma optimize( "", off)

namespace AT_NAME
{
	static inline AT_DEVICE_MTRL_API real computeFlakeOrentationDistribution(
		const aten::vec4& flake,
		real flake_normal_orientation,
		const aten::vec3& normal)
	{
		const auto delta = flake_normal_orientation;
		const auto cosbeta = dot((aten::vec3)flake, normal);
		const auto p = real(1) / (2 * AT_MATH_PI * delta) * aten::exp((cosbeta - 1) / delta);
		return p;
	}

	AT_DEVICE_MTRL_API real CarPaintBRDF::pdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		const auto ni = real(1);
		const auto nt = param->ior;

		aten::vec3 V = -wi;
		aten::vec3 L = wo;
		aten::vec3 N = normal;
		aten::vec3 H = normalize(L + V);

		real F(1);
		{
			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (ni - nt) / (ni + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(L, H));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		real pdf = real(0);

		{
			const auto m = param->clearcoatRoughness;

			const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

			// half vector.
			auto wh = normalize(-wi + wo);

			auto costheta = dot(normal, wh);

			auto c = dot(wo, wh);

			pdf += F * (n + 1) / (2 * AT_MATH_PI) * aten::pow(costheta, n) / (4 * c);
		}

		auto density = FlakesNormal::computeFlakeDensity(param->flake_size, real(1280) / 720);

#if 1
		{
			auto flakeNml = FlakesNormal::gen(
				u, v,
				param->flake_scale,
				param->flake_size,
				param->flake_size_variance,
				param->flake_normal_orientation);

			const auto p = computeFlakeOrentationDistribution(flakeNml, param->flake_normal_orientation, aten::vec3(0, 0, 1));
			pdf += density * p;
		}
#endif	

		{
			auto c = dot(normal, wo);
			c = aten::abs(c);

			pdf += (1 - F) * (1 - density) * c / AT_MATH_PI;
		}

		return pdf;
	}

	AT_DEVICE_MTRL_API real CarPaintBRDF::pdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return pdf(&m_param, normal, wi, wo, u, v);
	}

	static inline AT_DEVICE_MTRL_API aten::vec3 computeRefract(
		real nc,
		real nt,
		const aten::vec3& wi,
		const aten::vec3& normal)
	{
		aten::vec3 in = -wi;
		aten::vec3 nml = normal;

		bool into = (dot(in, normal) > real(0));

		if (!into) {
			nml = -nml;
		}

		real nnt = into ? nc / nt : nt / nc;
		real ddn = dot(wi, nml);

		// NOTE
		// https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5

		auto d = dot(in, nml);
		auto refract = -nnt * (in - d * nml) - aten::sqrt(real(1) - nnt * nnt * (1 - ddn * ddn)) * nml;

		return std::move(refract);
	}

	AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::sampleDirection(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		const auto ni = real(1);
		const auto nt = param->ior;

		const auto m = param->clearcoatRoughness;

		const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		const auto theta = aten::acos(aten::pow(r1, 1 / (n + 1)));
		const auto phi = 2 * AT_MATH_PI * r2;

		auto N = normal;
		auto T = aten::getOrthoVector(N);
		auto B = cross(N, T);

		auto costheta = aten::cos(theta);
		auto sintheta = aten::sqrt(1 - costheta * costheta);

		auto cosphi = aten::cos(phi);
		auto sinphi = aten::sqrt(1 - cosphi * cosphi);

		auto w = T * sintheta * cosphi + B * sintheta * sinphi + N * costheta;
		w = normalize(w);

		auto dir = wi - 2 * dot(wi, w) * w;

		real F(1);
		{
			aten::vec3 V = -wi;
			aten::vec3 L = dir;
			aten::vec3 N = normal;
			aten::vec3 H = normalize(L + V);

			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (ni - nt) / (ni + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(L, H));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		if (r1 < F) {
			// Nothing...
		}
		else {
			r1 -= F;
			r1 /= (1 - F);

			auto N = normal;
			auto T = aten::getOrthoVector(N);
			auto B = cross(N, T);

			// コサイン項を使った重点的サンプリング.
			r1 = 2 * AT_MATH_PI * r1;
			auto r2s = aten::sqrt(r2);

			const real x = aten::cos(r1) * r2s;
			const real y = aten::sin(r1) * r2s;
			const real z = aten::sqrt(real(1) - r2);

			dir = normalize((T * x + B * y + N * z));

			dir = computeRefract(ni, nt, dir, normal);
		}

		return std::move(dir);
	}

	AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
	}

	AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		auto albedo = param->baseColor;
		albedo *= material::sampleTexture(param->albedoMap, u, v, real(1));

		aten::vec3 ret = bsdf(param, normal, wi, wo, u, v, albedo);
		return std::move(ret);
	}

	AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v,
		const aten::vec3& externalAlbedo)
	{
		const auto ni = real(1);
		const auto nt = param->ior;

		const auto m = param->clearcoatRoughness;
		const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

		aten::vec3 V = -wi;
		aten::vec3 L = wo;
		aten::vec3 N = normal;
		aten::vec3 H = normalize(L + V);

		auto NdotH = aten::abs(dot(N, H));
		auto VdotH = aten::abs(dot(V, H));
		auto NdotL = aten::abs(dot(N, L));
		auto NdotV = aten::abs(dot(N, V));

		real F(1);
		{
			// http://d.hatena.ne.jp/hanecci/20130525/p3

			// NOTE
			// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
			// R0 = ((n1 - n2) / (n1 + n2))^2

			auto r0 = (ni - nt) / (ni + nt);
			r0 = r0 * r0;

			auto LdotH = aten::abs(dot(L, H));

			F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
		}

		aten::vec3 bsdf(0);

		// Compute G.
		real G(1);
		{
			auto G1_lh = MicrofacetGGX::computeGGXSmithG1(m, L, N);
			auto G1_vh = MicrofacetGGX::computeGGXSmithG1(m, V, N);

			G = G1_lh * G1_vh;
		}

		{
			real D = (n + 1) / (2 * AT_MATH_PI) * aten::pow(NdotH, n);

			auto denom = 4 * NdotL * NdotV;

			bsdf += denom > AT_MATH_EPSILON ? F * G * D / denom : 0;
		}

		const auto density = FlakesNormal::computeFlakeDensity(param->flake_size, real(1280) / 720);

#if 1
		{
			const auto D = density;
			const auto S = param->flake_size;
			const auto h = real(1);	// TODO
			const auto r = param->flake_reflection;
			const auto t = param->flake_transmittance;

			const auto tau = D * S * (1 - t);

			const auto Reff = ((real(1) - aten::exp(-2 * tau * h)) / (2 * tau * h)) * D * S * h * r;

			auto costheta = NdotV;
			auto sintheta = real(1) - costheta * costheta;

			auto _theta = aten::asin(ni / nt * sintheta);
			auto _costheta = aten::cos(_theta);

			auto denom = 4 * nt * nt * _costheta;

			auto flakeNml = FlakesNormal::gen(
				u, v,
				param->flake_scale,
				param->flake_size,
				param->flake_size_variance,
				param->flake_normal_orientation);

			const auto p = computeFlakeOrentationDistribution(flakeNml, param->flake_normal_orientation, aten::vec3(0, 0, 1));

			bsdf += denom > 0 ? Reff / denom * p : 0;
		}
#endif

		{
			bsdf += (1 - F) * (1 - density) * externalAlbedo / AT_MATH_PI;
		}

		return std::move(bsdf);
	}

	AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(&m_param, normal, wi, wo, u, v));
	}

	AT_DEVICE_MTRL_API MaterialSampling CarPaintBRDF::sample(
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

	AT_DEVICE_MTRL_API void CarPaintBRDF::sample(
		MaterialSampling* result,
		const aten::MaterialParameter* param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		auto albedo = param->baseColor;
		albedo *= material::sampleTexture(param->albedoMap, u, v, real(1));

		sample(
			result,
			param,
			normal,
			wi,
			orgnormal,
			sampler,
			u, v,
			albedo,
			isLightPath);
	}

	AT_DEVICE_MTRL_API void CarPaintBRDF::sample(
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
		result->dir = sampleDirection(param, normal, wi, u, v, sampler);
		
		const auto& wo = result->dir;

		result->pdf = pdf(param, normal, wi, wo, u, v);
		result->bsdf = bsdf(param, normal, wi, wo, u, v, externalAlbedo);
	}

	bool CarPaintBRDF::edit(aten::IMaterialParamEditor* editor)
	{
		bool b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, clearcoatRoughness, 0, 1);

		bool b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_scale, 1, 1000);
		bool b2 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_size, 0.01, 1);
		bool b3 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_size_variance, 0, 1);
		bool b4 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_normal_orientation, 0, 1);
		
		bool b5 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_reflection, 0, 1);
		bool b6 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, flake_transmittance, 0, 1);

		bool b7 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
		bool b8 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

		return b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8;
	}
}
