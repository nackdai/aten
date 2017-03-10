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
			: material(albedo, albedoMap, normalMap), m_shininess(shininess), m_ior(ior)
		{}

		MicrofacetBlinn(Values& val)
			: material(val)
		{
			m_ior = val.get("ior", m_ior);
			m_shininess = val.get("shininess", m_shininess);
		}

		virtual ~MicrofacetBlinn() {}

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
		real m_shininess{ real(0) };

		// ï®ëÃÇÃã¸ê‹ó¶.
		real m_ior;
	};
}
