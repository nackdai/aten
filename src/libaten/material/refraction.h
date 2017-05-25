#pragma once

#include "material/material.h"

namespace AT_NAME
{
	class refraction : public material {
	public:
		refraction(
			const aten::vec3& albedo,
			real ior,
			bool isIdealRefraction = false,
			aten::texture* normalMap = nullptr)
			: material(aten::MaterialType::Refraction, MaterialAttributeRefraction, albedo, ior, nullptr, normalMap)
		{
			m_param.isIdealRefraction = isIdealRefraction;
		}

		refraction(aten::Values& val)
			: material(aten::MaterialType::Refraction, MaterialAttributeRefraction, val)
		{
			m_param.isIdealRefraction = val.get("isIdealRefraction", m_param.isIdealRefraction);
		}

		virtual ~refraction() {}

	public:
		void setIsIdealRefraction(bool f)
		{
			m_param.isIdealRefraction = f;
		}
		bool isIdealRefraction() const
		{
			return (m_param.isIdealRefraction > 0);
		}

		static AT_DEVICE_API real pdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v);

		static AT_DEVICE_API aten::vec3 sampleDirection(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			real u, real v,
			aten::sampler* sampler);

		static AT_DEVICE_API aten::vec3 bsdf(
			const aten::MaterialParameter* param,
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
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

		struct RefractionSampling {
			bool isRefraction;
			bool isIdealRefraction;
			real probReflection;
			real probRefraction;

			RefractionSampling(bool _isRefraction, real _probReflection, real _probRefraction, bool _isIdealRefraction = false)
				: isRefraction(_isRefraction), probReflection(_probReflection), probRefraction(_probRefraction),
				isIdealRefraction(_isIdealRefraction)
			{}
		};

		static RefractionSampling check(
			material* mtrl,
			const aten::vec3& in,
			const aten::vec3& normal,
			const aten::vec3& orienting_normal);

		virtual real computeFresnel(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor = 1) const override final
		{
			// TODO
			AT_ASSERT(false);
			return real(0);
		}
	};
}
