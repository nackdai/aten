#pragma once

#include "light/light.h"

namespace aten {
	class AreaLight : public Light {
	public:
		AreaLight() {}
		AreaLight(hitable* obj, const vec3& le)
		{
			m_object = obj;
			m_le = le;
		}

		virtual ~AreaLight() {}

	public:
		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			if (m_object) {
				hitrecord rec;

				vec3 pos;
				if (sampler) {
					result.r1 = sampler->nextSample();
					result.r2 = sampler->nextSample();

					pos = m_object->getRandomPosOn(result.r1, result.r2);
				}
				else {
					pos = m_object->getBoundingbox().getCenter();
				}

				auto dir = pos - org;
				auto dist = dir.length();

				ray r(
					org,
					normalize(dir));

				bool isHit = m_object->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					/*if (aten::abs(dist - rec.t) < AT_MATH_EPSILON)*/ {
						result.pos = pos;
						result.pdf = 1 / rec.area;
						result.dir = rec.p - org;
						result.nml = rec.normal;
						
						result.le = m_le;
						result.intensity = 1;
						result.finalColor = m_le;

						result.obj = m_object;

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
			return m_object;
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final
		{
			bool isHit = false;

			if (m_object) {
				isHit = m_object->hit(r, t_min, t_max, rec);
			}

			return isHit;
		}

		virtual aabb getBoundingbox() const override final
		{
			if (m_object) {
				auto box = m_object->getBoundingbox();
				return std::move(box);
			}

			return std::move(aabb());
		}

	private:
		hitable* m_object{ nullptr };
	};
}