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
			bool isIdealRefraction = false,
			texture* normalMap = nullptr)
			: material(albedo, ior, nullptr, normalMap)
		{
			m_isIdealRefraction = isIdealRefraction;
		}

		refraction(Values& val)
			: material(val)
		{
			m_isIdealRefraction = val.get("isIdealRefraction", m_isIdealRefraction);
		}

		virtual ~refraction() {}

	public:
		virtual bool isSingular() const override final
		{
			return true;
		}

		virtual bool isTranslucent() const override final
		{
			return true;
		}

		bool setIsIdealRefraction(bool f)
		{
			m_isIdealRefraction = f;
		}
		bool isIdealRefraction() const
		{
			return m_isIdealRefraction;
		}

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v,
			sampler* sampler) const override final;

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
			real u, real v) const override final;

		virtual real computeFresnel(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real outsideIor = 1) const override final
		{
			// TODO
			AT_ASSERT(false);
			return real(0);
		}

	private:
		bool m_isIdealRefraction{ false };
	};
}
