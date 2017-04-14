#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetBeckman : public material {
	public:
		MicrofacetBeckman() {}
		MicrofacetBeckman(
			const vec3& albedo,
			real roughness, real ior,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(albedo, ior, albedoMap, normalMap), m_roughnessMap(roughnessMap)
		{
			m_roughness = aten::clamp<real>(roughness, 0, 1);
		}

		MicrofacetBeckman(Values& val)
			: material(val)
		{
			m_roughness = val.get("roughness", m_roughness);
			m_roughness = aten::clamp<real>(m_roughness, 0, 1);

			m_roughnessMap = val.get("roughnessmap", m_roughnessMap);
		}

		virtual ~MicrofacetBeckman() {}

	public:
		virtual bool isGlossy() const override final
		{
			return (m_roughness == 1 ? false : true);
		}

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

		virtual sampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

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
