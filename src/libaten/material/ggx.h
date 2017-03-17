#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetGGX : public material {
	public:
		MicrofacetGGX() {}
		MicrofacetGGX(
			const vec3& albedo,
			real roughness, real ior,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(albedo, ior, albedoMap, normalMap), m_roughnessMap(roughnessMap)
		{
			m_roughness = aten::clamp<real>(roughness, 0, 1);
		}

		MicrofacetGGX(Values& val)
			: material(val)
		{
			m_ior = val.get("ior", m_ior);
			m_roughness = val.get("roughness", m_roughness);
			m_roughnessMap = val.get("roughnessmap", m_roughnessMap);
		}

		virtual ~MicrofacetGGX() {}

	public:
		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal, 
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual sampling sample(
			const vec3& in,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const override final;

	private:
		inline real sampleRoughness(real u, real v) const;

		real pdf(
			real roughness,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo) const;

		vec3 sampleDirection(
			real roughness,
			const vec3& in,
			const vec3& normal,
			sampler* sampler) const;

		vec3 bsdf(
			real roughness,
			real& fresnel,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v) const;

	private:
		real m_roughness{ real(0) };

		texture* m_roughnessMap{ nullptr };
	};
}
