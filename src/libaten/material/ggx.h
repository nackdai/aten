#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class MicrofacetGGX : public material {
	public:
		MicrofacetGGX(
			const aten::vec3& albedo = aten::vec3(0.5),
			real roughness = real(0.5), 
			real ior = real(1),
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr,
			aten::texture* roughnessMap = nullptr)
			: material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, albedo, ior, albedoMap, normalMap)
		{
			m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
			m_param.roughness = aten::clamp<real>(roughness, 0, 1);
		}

		MicrofacetGGX(aten::Values& val)
			: material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, val)
		{
			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

			auto roughnessMap = (aten::texture*)val.get("roughnessmap", nullptr);
			m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
		}

		virtual ~MicrofacetGGX() {}

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

		virtual bool edit(aten::IMaterialParamEditor* editor) override final
		{
			auto b0 = AT_EDIT_MATERIAL_PARAM(editor, m_param, roughness);
			auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
			auto b2 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

			AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
			AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);
			AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, roughnessMap);

			return b0 || b1 || b2;
		}

	private:
		static AT_DEVICE_MTRL_API real pdf(
			const real roughness,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo);

		static AT_DEVICE_MTRL_API aten::vec3 sampleDirection(
			const real roughness,
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
