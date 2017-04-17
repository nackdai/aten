#pragma once

#include <vector>
#include "material/material.h"

namespace aten
{
	class LayeredBSDF : public material {
	public:
		LayeredBSDF() {}
		virtual ~LayeredBSDF() {}

	public:
		void add(material* mtrl);

		virtual vec3 sampleAlbedoMap(real u, real v) const override final;

		virtual bool isGlossy() const override final;

		virtual void applyNormalMap(
			const vec3& orgNml,
			vec3& newNml,
			real u, real v) const override final;

		virtual real computeFresnel(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real outsideIor = 1) const override final;

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
		std::vector<material*> m_layer;
	};
}
