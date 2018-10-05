#pragma once

#include "material/material.h"

namespace AT_NAME
{
	// NOTE
	// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
	// https://agraphicsguy.wordpress.com/2015/11/11/glass-material-simulated-by-microfacet-bxdf/

	class MicrofacetRefraction : public material {
	public:
		MicrofacetRefraction(
			const aten::vec3& albedo = aten::vec3(0.5),
			real roughness = real(0.5),
			real ior = real(1),
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr,
			aten::texture* roughnessMap = nullptr)
			: material(aten::MaterialType::Microfacet_Refraction, MaterialAttributeRefraction, albedo, ior, nullptr, normalMap)
		{
		}

		MicrofacetRefraction(aten::Values& val)
			: material(aten::MaterialType::Microfacet_Refraction, MaterialAttributeRefraction, val)
		{
		}

	public:
		static AT_DEVICE_MTRL_API real pdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v,
			const aten::vec3& externalAlbedo);

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false);

		static AT_DEVICE_MTRL_API void sample(
			MaterialSampling* result,
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			const aten::vec3& externalAlbedo,
			bool isLightPath = false);

		virtual AT_DEVICE_MTRL_API real pdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal, 
			real u, real v,
			aten::sampler* sampler) const override final;

		virtual AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::vec3& normal, 
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const override final;

		virtual AT_DEVICE_MTRL_API MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::vec3& orgnormal,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

		virtual bool edit(aten::IMaterialParamEditor* editor) override final;

	private:
		static AT_DEVICE_MTRL_API real pdf(
			const real roughness,
			real ior,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo);

		static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const real roughness,
			real ior,
			const aten::vec3& in,
			const aten::vec3& normal,
			aten::sampler* sampler);

		static AT_DEVICE_MTRL_API aten::vec3 bsdf(
			const aten::vec3& albedo,
			const real roughness,
			const real ior,
			real& fresnel,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);
	};
}
