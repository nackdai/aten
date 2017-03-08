#pragma once

#include "material/material.h"

namespace aten
{
	class refraction : public material {
	public:
		refraction() {}
		refraction(
			const vec3& albedo,
			real ior,
			texture* normalMap = nullptr)
			: material(albedo, nullptr, normalMap), m_nt(ior)
		{}

		virtual ~refraction() {}

		virtual bool isSingular() const override final
		{
			return true;
		}

		virtual bool isTranslucent() const override final
		{
			return true;
		}

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
		// ï®ëÃÇÃã¸ê‹ó¶.
		real m_nt;
	};
}
