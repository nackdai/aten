#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten
{
	class lambert : public material {
	public:
		lambert() {}
		lambert(
			const vec3& albedo, 
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr)
			: material(albedo, albedoMap, normalMap)
		{}

		virtual ~lambert() {}

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
	};
}
