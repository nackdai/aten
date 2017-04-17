#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetBlinn : public material {
	public:
		MicrofacetBlinn() {}
		MicrofacetBlinn(
			const vec3& albedo,
			real shininess, real ior,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr)
			: material(albedo, ior, albedoMap, normalMap)
		{
			m_param.shininess = shininess;
		}

		MicrofacetBlinn(Values& val)
			: material(val)
		{
			m_param.shininess = val.get("shininess", m_param.shininess);
		}

		virtual ~MicrofacetBlinn() {}

	public:
		virtual bool isGlossy() const override final
		{
			return (m_param.shininess == 0 ? false : true);
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
		vec3 bsdf(
			real& fresnel,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v) const;
	};
}
