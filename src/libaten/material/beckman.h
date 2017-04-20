#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetBeckman : public material {
	public:
		MicrofacetBeckman(
			const vec3& albedo,
			real roughness, real ior,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.roughnessMap.ptr = roughnessMap;
			m_param.roughness = aten::clamp<real>(roughness, 0, 1);
		}

		MicrofacetBeckman(Values& val)
			: material(MaterialAttributeMicrofacet, val)
		{
			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

			m_param.roughnessMap = val.get("roughnessmap", m_param.roughnessMap);
		}

		virtual ~MicrofacetBeckman() {}

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
		static real pdf(
			const real roughness,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo);

		static vec3 sampleDirection(
			const real roughness,
			const vec3& in,
			const vec3& normal,
			sampler* sampler);

		static vec3 bsdf(
			const vec3& albedo,
			const real roughness,
			const real ior,
			real& fresnel,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);
	};
}
