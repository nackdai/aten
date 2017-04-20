#pragma once

#include "light/light.h"

namespace aten {
	class DirectionalLight : public Light {
	public:
		DirectionalLight()
			: Light(LightAttributeDirectional)
		{}
		DirectionalLight(
			const vec3& dir,
			const vec3& le)
			: Light(LightAttributeDirectional)
		{
			m_param.dir = normalize(dir);
			m_param.le = le;
		}

		DirectionalLight(Values& val)
			: Light(LightAttributeDirectional, val)
		{}

		virtual ~DirectionalLight() {}

	public:
		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			return std::move(sample(m_param, org, sampler));
		}

		static LightSampleResult sample(
			const LightParameter& param,
			const vec3& org,
			sampler* sampler)
		{
			LightSampleResult result;

			result.pdf = real(1);
			result.dir = -normalize(param.dir);
			result.nml = vec3();	// Not used...

			result.le = param.le;
			result.intensity = real(1);
			result.finalColor = param.le;

			return std::move(result);
		}
	};
}