#pragma once

#include "light/light.h"

namespace aten {
	class DirectionalLight : public Light {
	public:
		DirectionalLight() {}
		DirectionalLight(
			const vec3& dir,
			const vec3& le)
		{
			m_param.dir = normalize(dir);
			m_param.le = le;
		}

		DirectionalLight(Values& val)
			: Light(val)
		{}

		virtual ~DirectionalLight() {}

	public:
		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			result.pdf = real(1);
			result.dir = -normalize(m_param.dir);
			result.nml = vec3();	// Not used...

			result.le = m_param.le;
			result.intensity = real(1);
			result.finalColor = m_param.le;

			return std::move(result);
		}

		virtual bool isInifinite() const override final
		{
			return true;
		}
	};
}