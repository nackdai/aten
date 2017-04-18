#pragma once

#include "light/light.h"

namespace aten {
	class AreaLight : public Light {
	public:
		AreaLight() 
			: Light(LightTypeArea)
		{}
		AreaLight(hitable* obj, const vec3& le)
			: Light(LightTypeArea)
		{
			m_param.object.ptr = obj;
			m_param.le = le;
		}

		AreaLight(Values& val)
			: Light(LightTypeArea, val)
		{
			m_param.object.ptr = (hitable*)val.get("object", m_param.object.ptr);
			m_param.le = val.get("color", m_param.le);
		}

		virtual ~AreaLight() {}

	public:
		static LightSampleResult sample(
			const LightParameter& param,
			const vec3& org,
			sampler* sampler);

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final;

		virtual const hitable* getLightObject() const override final
		{
			return (hitable*)m_param.object.ptr;
		}

		virtual hitable::SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override final
		{
			if (m_param.object.ptr) {
				auto obj = getLightObject();
				return obj->getSamplePosNormalPdf(sampler);
			}
			return std::move(hitable::SamplingPosNormalPdf(vec3(0), vec3(1), real(0)));
		}
	};
}