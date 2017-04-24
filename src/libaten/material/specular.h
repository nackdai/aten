#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class specular : public material {
	public:
		specular(
			const aten::vec3& albedo,
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr)
			: material(aten::MaterialType::Specular, MaterialAttributeSpecular, albedo, 0, albedoMap, normalMap)
		{}

		specular(aten::Values& val)
			: material(aten::MaterialType::Specular, MaterialAttributeSpecular, val)
		{}

		virtual ~specular() {}

	public:
		static AT_DEVICE_API real pdf(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API MaterialSampling sample(
			const aten::MaterialParameter& param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::hitrecord& hitrec,
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
			const aten::hitrecord& hitrec,
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
