#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten
{
	class lambert : public material {
	public:
		lambert(
			const vec3& albedo, 
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr)
			: material(MaterialTypeLambert, albedo, 0, albedoMap, normalMap)
		{}

		lambert(Values& val)
			: material(MaterialTypeLambert, val)
		{}

		virtual ~lambert() {}

	public:
		static real pdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);

		static real pdf(
			const vec3& normal,
			const vec3& wo);

		static vec3 sampleDirection(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			real u, real v,
			sampler* sampler);

		static vec3 sampleDirection(
			const vec3& normal,
			sampler* sampler);

		static vec3 bsdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);

		static vec3 bsdf(
			const MaterialParameter& param,
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
