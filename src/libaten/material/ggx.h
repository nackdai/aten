#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class MicrofacetGGX : public material {
	public:
		MicrofacetGGX(
			const aten::vec3& albedo,
			real roughness, real ior,
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr,
			aten::texture* roughnessMap = nullptr)
			: material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.roughnessMap.ptr = roughnessMap;
			m_param.roughness = aten::clamp<real>(roughness, 0, 1);
		}

		MicrofacetGGX(aten::Values& val)
			: material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, val)
		{
			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

			m_param.roughnessMap.ptr = val.get("roughnessmap", m_param.roughnessMap.ptr);
		}

		virtual ~MicrofacetGGX() {}

	public:
		static AT_DEVICE_API real pdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API MaterialSampling sample(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::hitrecord& hitrec,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false);

		virtual real pdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal, 
			real u, real v,
			aten::sampler* sampler) const override final;

		virtual aten::vec3 bsdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::hitrecord& hitrec,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

	private:
		static AT_DEVICE_API real pdf(
			const real roughness,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const real roughness,
			const aten::vec3& in,
			const aten::vec3& normal,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::vec3& albedo,
			const real roughness,
			const real ior,
			real& fresnel,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);
	};
}
