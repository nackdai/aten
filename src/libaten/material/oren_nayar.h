#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace AT_NAME
{
	class OrenNayar : public material {
	public:
		OrenNayar(
			const aten::vec3& albedo = aten::vec3(0.5),
			real roughness = real(0.5),
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr,
			aten::texture* roughnessMap = nullptr)
			: material(aten::MaterialType::OrneNayar, MaterialAttributeLambert, albedo, 1, albedoMap, normalMap)
		{
			m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
			m_param.roughness = aten::clamp<real>(roughness, 0, 1);
		}

		OrenNayar(aten::Values& val)
			: material(aten::MaterialType::OrneNayar, MaterialAttributeLambert, val)
		{
			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

			auto roughnessMap = (aten::texture*)val.get("roughnessmap", nullptr);
			m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
		}

		virtual ~OrenNayar() {}

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
	};
}
