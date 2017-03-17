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

		virtual real pdf(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v,
			sampler* sampler) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return real(1);
		}

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal, 
			real u, real v,
			sampler* sampler) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return std::move(normal);
		}

		virtual vec3 bsdf(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return std::move(vec3());
		}

		virtual sampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const override final
		{
			AT_ASSERT(false);
			return std::move(sampling(vec3(), vec3(), real(0)));
		}
	};
}
