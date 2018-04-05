#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class CarPaintBRDF : public material {
	public:
		CarPaintBRDF(
			const aten::vec3& albedo = aten::vec3(0.5),
			real ior = real(1),
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr,
			aten::texture* flakesMap = nullptr)
			: material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.roughnessMap = flakesMap ? flakesMap->id() : -1;

			m_param.clearcoatRoughness = real(0.5);

			m_param.flake_scale = real(100);
			m_param.flake_size = real(0.01);
			m_param.flake_size_variance = real(0.25);
			m_param.flake_normal_orientation = real(0.5);
			
			m_param.flake_reflection = real(0.5);
			m_param.flake_transmittance = real(0.5);

			m_param.thicknessPaintLayer = real(1);
		}

		CarPaintBRDF(aten::Values& val)
			: material(aten::MaterialType::CarPaint, MaterialAttributeMicrofacet, val)
		{
			// TODO

			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

			auto roughnessMap = (aten::texture*)val.get("roughnessmap", nullptr);
			m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
		}

		virtual ~CarPaintBRDF() {}

	public:
		static AT_DEVICE_MTRL_API real pdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v,
			const aten::vec3& externalAlbedo);

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false);

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			const aten::vec3& externalAlbedo,
			bool isLightPath = false);

		virtual AT_DEVICE_MTRL_API real pdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal, 
			real u, real v,
			aten::sampler* sampler) const override final;

		virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual AT_DEVICE_MTRL_API MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

		virtual bool edit(aten::IMaterialParamEditor* editor) override final;
	};
}
