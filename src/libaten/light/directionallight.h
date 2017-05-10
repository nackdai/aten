#pragma once

#include "light/light.h"

namespace AT_NAME {
	class DirectionalLight : public Light {
	public:
		DirectionalLight()
			: Light(aten::LightType::Direction, LightAttributeDirectional)
		{}
		DirectionalLight(
			const aten::vec3& dir,
			const aten::vec3& le)
			: Light(aten::LightType::Direction, LightAttributeDirectional)
		{
			m_param.dir = aten::normalize(dir);
			m_param.le = le;
		}

		DirectionalLight(aten::Values& val)
			: Light(aten::LightType::Direction, LightAttributeDirectional, val)
		{}

		virtual ~DirectionalLight() {}

	public:
		virtual aten::LightSampleResult sample(const aten::vec3& org, aten::sampler* sampler) const override final
		{
			return std::move(sample(&m_param, org, sampler));
		}

		static AT_DEVICE_API aten::LightSampleResult sample(
			const aten::LightParameter* param,
			const aten::vec3& org,
			aten::sampler* sampler)
		{
			aten::LightSampleResult result;

			result.pdf = real(1);
			result.dir = -normalize(param->dir);
			result.nml = aten::vec3();	// Not used...

			result.le = param->le;
			result.intensity = real(1);
			result.finalColor = param->le;

			return std::move(result);
		}
	};
}