#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace AT_NAME
{
	class lambert : public material {
	public:
		lambert(
			const aten::vec3& albedo, 
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr)
			: material(aten::MaterialType::Lambert, MaterialAttributeLambert, albedo, 0, albedoMap, normalMap)
		{}

		lambert(aten::Values& val)
			: material(aten::MaterialType::Lambert, MaterialAttributeLambert, val)
		{}

		virtual ~lambert() {}

	public:
		static AT_DEVICE_API real pdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API real pdf(
			const aten::vec3& normal,
			const aten::vec3& wo);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::vec3& normal,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			real u, real v);

		static AT_DEVICE_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false);

		virtual real pdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal,
			real u, real v,
			aten::sampler* sampler) const override final;

		virtual aten::vec3 bsdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

		virtual real computeFresnel(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor = 1) const override final
		{
			return real(1);
		}
	};
}
