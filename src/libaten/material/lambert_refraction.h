#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace AT_NAME
{
	class LambertRefraction : public material {
	public:
		LambertRefraction(
			const aten::vec3& albedo = aten::vec3(0.5),
			real ior = real(1),
			aten::texture* normalMap = nullptr)
			: material(aten::MaterialType::Lambert_Refraction, MaterialAttributeTransmission, albedo, ior, nullptr, normalMap)
		{}

		LambertRefraction(aten::Values& val)
			: material(aten::MaterialType::Lambert_Refraction, MaterialAttributeTransmission, val)
		{}

		virtual ~LambertRefraction() {}

	public:
		static AT_DEVICE_MTRL_API real pdf(
			const aten::vec3& normal,
			const aten::vec3& wo)
		{
			auto c = dot(normal, wo);
			//AT_ASSERT(c >= 0);
			c = aten::abs(c);

			auto ret = c / AT_MATH_PI;

			return ret;
		}

		static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler)
		{
			aten::vec3 in = -wi;
			aten::vec3 nml = normal;

			bool into = (dot(in, normal) > real(0));

			if (!into) {
				nml = -nml;
			}

			// normalの方向を基準とした正規直交基底(w, u, v)を作る.
			// この基底に対する半球内で次のレイを飛ばす.
			auto n = nml;
			auto t = aten::getOrthoVector(n);
			auto b = cross(n, t);

			// コサイン項を使った重点的サンプリング.
			const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
			const real r2 = sampler->nextSample();
			const real r2s = sqrt(r2);

			const real x = aten::cos(r1) * r2s;
			const real y = aten::sin(r1) * r2s;
			const real z = aten::sqrt(real(1) - r2);

			aten::vec3 dir = normalize((t * x + b * y + n * z));
			//AT_ASSERT(dot(normal, dir) >= 0);

			return std::move(dir);
		}

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			real u, real v)
		{
			aten::vec3 albedo = param->baseColor;
			albedo *= sampleTexture(
				param->albedoMap,
				u, v,
				real(1));

			aten::vec3 ret = albedo / AT_MATH_PI;
			return ret;
		}

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& externalAlbedo)
		{
			aten::vec3 albedo = param->baseColor;
			albedo *= externalAlbedo;

			aten::vec3 ret = albedo / AT_MATH_PI;
			return ret;
		}

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false)
		{
			MaterialSampling ret;

			result->dir = sampleDirection(param, normal, wi, u, v, sampler);
			result->pdf = pdf(normal, result->dir);
			result->bsdf = bsdf(param, u, v);
		}

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			const aten::vec3& externalAlbedo,
			bool isLightPath = false)
		{
			MaterialSampling ret;

			result->dir = sampleDirection(param, normal, wi, real(0), real(0), sampler);
			result->pdf = pdf(normal, result->dir);
			result->bsdf = bsdf(param, externalAlbedo);
		}

		virtual AT_DEVICE_MTRL_API real pdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final
		{
			auto ret = pdf(normal, wo);
			return ret;
		}

		virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal,
			real u, real v,
			aten::sampler* sampler) const override final
		{
			return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
		}

		virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final
		{
			auto ret = bsdf(&m_param, u, v);
			return std::move(ret);
		}

		virtual AT_DEVICE_MTRL_API MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final
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

		virtual AT_DEVICE_MTRL_API real computeFresnel(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor = 1) const override final
		{
			return computeFresnel(&m_param, normal, wi, wo, outsideIor);
		}

		static AT_DEVICE_MTRL_API real computeFresnel(
			const aten::MaterialParameter* mtrl,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor)
		{
			return real(1);
		}

		virtual bool edit(aten::IMaterialParamEditor* editor) override final
		{
			AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
			AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

			auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
			auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

			return b0 || b1;
		}
	};
}
