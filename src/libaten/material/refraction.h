#pragma once

#include "material/material.h"

namespace aten
{
	class refraction : public material {
	public:
		refraction() {}
		refraction(const vec3& c, real nt)
			: m_color(c), m_nt(nt)
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

		virtual vec3 color() const override final
		{
			return m_color;
		}

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo) const override final;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal, 
			sampler* sampler) const override final;

		virtual vec3 brdf(
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
		vec3 m_color;

		// ï®ëÃÇÃã¸ê‹ó¶.
		real m_nt;
	};
}
