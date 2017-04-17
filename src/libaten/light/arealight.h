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
		virtual real samplePdf(const ray& r) const override final
		{
			real pdf = 0;

			auto obj = getLightObject();

			if (obj) {
				hitrecord rec;
				bool isHit = obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					pdf = 1 / rec.area;
				}
			}

			return pdf;
		}

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			auto obj = getLightObject();

			if (obj) {
				hitrecord rec;

				vec3 pos;
				if (sampler) {
					pos = obj->getRandomPosOn(sampler);
				}
				else {
					pos = obj->getBoundingbox().getCenter();
				}

				auto dir = pos - org;
				auto dist = dir.length();

				ray r(
					org,
					normalize(dir));

				bool isHit = obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					/*if (aten::abs(dist - rec.t) < AT_MATH_EPSILON)*/ {
						result.pos = pos;
						result.pdf = 1 / rec.area;
						result.dir = rec.p - org;
						result.nml = rec.normal;
						
						result.le = m_param.le;
						result.intensity = 1;
						result.finalColor = m_param.le;

						result.obj = const_cast<hitable*>(obj);

					}
				}
			}

			return std::move(result);
		}

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
			hitrecord& rec) const override final
		{
			bool isHit = false;

			if (m_param.object.ptr) {
				auto obj = getLightObject();
				isHit = obj->hit(r, t_min, t_max, rec);
			}

			return isHit;
		}

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