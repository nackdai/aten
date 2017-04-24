#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class MicrofacetBlinn : public material {
	public:
		MicrofacetBlinn(
			const aten::vec3& albedo,
			real shininess, real ior,
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr)
			: material(aten::MaterialType::Blinn, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.shininess = shininess;
		}

		MicrofacetBlinn(aten::Values& val)
			: material(aten::MaterialType::Blinn, MaterialAttributeMicrofacet, val)
		{
			m_param.shininess = val.get("shininess", m_param.shininess);
		}

		virtual ~MicrofacetBlinn() {}

	public:
		static AT_DEVICE_API real pdf(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API MaterialSampling sample(
			const aten::MaterialParameter& param,
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
		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::vec3& albedo,
			const real shininess,
			const real ior,
			real& fresnel,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);
	};
}
