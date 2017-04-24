#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetBlinn : public material {
	public:
		MicrofacetBlinn(
			const vec3& albedo,
			real shininess, real ior,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr)
			: material(MaterialType::MicrofacetBlinn, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.shininess = shininess;
		}

		MicrofacetBlinn(Values& val)
			: material(MaterialType::MicrofacetBlinn, MaterialAttributeMicrofacet, val)
		{
			m_param.shininess = val.get("shininess", m_param.shininess);
		}

		virtual ~MicrofacetBlinn() {}

	public:
		static real pdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);

		static vec3 sampleDirection(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			real u, real v,
			sampler* sampler);

		static vec3 bsdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);

		static MaterialSampling sample(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false);

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal, 
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual MaterialSampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

	private:
		static vec3 bsdf(
			const vec3& albedo,
			const real shininess,
			const real ior,
			real& fresnel,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);
	};
}
