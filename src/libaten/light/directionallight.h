#pragma once

#include "light/light.h"

namespace aten {
	class DirectionalLight : public Light {
	public:
		DirectionalLight() {}
		DirectionalLight(
			const vec3& pos,
			const vec3& intensity)
		{
			m_pos = pos;
			m_intensity = intensity;
		}

		virtual ~DirectionalLight() {}

	public:
		virtual real getPdf(const vec3& org, sampler* sampler) const override final
		{
			return real(1);
		}

		virtual vec3 sampleDirToLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 dir = AT_MATH_INF * 0.5 * -m_dir;
			return std::move(dir);
		}

		virtual vec3 sampleNormalOnLight(const vec3& org, sampler* sampler) const override final
		{
			// Do not use...
			return std::move(m_dir);
		}

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			result.pdf = getPdf(org, sampler);
			result.dir = sampleDirToLight(org, sampler);
			result.nml = sampleNormalOnLight(org, sampler);
			result.intensity = m_intensity;

			return std::move(result);
		}
	};
}