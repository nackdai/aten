#pragma once

#include "material/material.h"

namespace aten
{
	class emissive : public material {
	public:
		emissive() {}
		emissive(const vec3& e)
			: material(e)
		{}

		emissive(Values& val)
			: material(val)
		{}

		virtual ~emissive() {}

		virtual bool isEmissive() const override final
		{
			return true;
		}

		virtual bool isGlossy() const override final
		{
			return false;
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

		virtual void serialize(MaterialParam& param) const override final;
	};
}
