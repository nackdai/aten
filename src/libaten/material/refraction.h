#pragma once

#include "material/material.h"

namespace aten
{
	class refraction : public material {
	public:
		refraction(
			const vec3& albedo,
			real ior,
			bool isIdealRefraction = false,
			texture* normalMap = nullptr)
			: material(MaterialAttributeRefraction, albedo, ior, nullptr, normalMap)
		{
			m_param.isIdealRefraction = isIdealRefraction;
		}

		refraction(Values& val)
			: material(MaterialAttributeRefraction, val)
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
			return m_param.isIdealRefraction;
		}

		static real pdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v);

		static vec3 sampleDirection(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			real u, real v,
			sampler* sampler);

		static vec3 bsdf(
			const MaterialParameter& param,
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
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
			const vec3& in,
			const vec3& normal,
			const vec3& orienting_normal);

		virtual real computeFresnel(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real outsideIor = 1) const override final
		{
			// TODO
			AT_ASSERT(false);
			return real(0);
		}
	};
}
