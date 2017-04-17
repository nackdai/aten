#pragma once

#include "light/light.h"

namespace aten {
	class AreaLight : public Light {
	public:
		AreaLight() {}
		AreaLight(hitable* obj, const vec3& le)
		{
			m_param.object.ptr = obj;
			m_param.le = le;
		}

		AreaLight(Values& val)
			: Light(val)
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

		static bool hit(
			const LightParameter& param,
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final;

		virtual bool isSingular() const override final
		{
			return false;
		}

		virtual const hitable* getLightObject() const override final
		{
			return (hitable*)m_param.object.ptr;
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;
		
		virtual aabb getBoundingbox() const override final
		{
			if (m_param.object.ptr) {
				auto obj = getLightObject();
				auto box = obj->getBoundingbox();
				return std::move(box);
			}

			return std::move(aabb());
		}

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const
		{
			if (m_param.object.ptr) {
				auto obj = getLightObject();
				return obj->getSamplePosNormalPdf(sampler);
			}
			return SamplingPosNormalPdf(vec3(0), vec3(1), real(0));
		}
	};
}