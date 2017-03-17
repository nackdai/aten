#pragma once

#include "material/material.h"

namespace aten
{
	class specular : public material {
	public:
		specular() {}
		specular(
			const vec3& albedo,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr)
			: material(albedo, 0, albedoMap, normalMap)
		{}

		specular(Values& val)
			: material(val)
		{}

		virtual ~specular() {}

		virtual bool isSingular() const override final
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
			return real(1);
		}
	};
}
