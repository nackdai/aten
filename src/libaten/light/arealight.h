#pragma once

#include "light/light.h"

namespace aten {
	class AreaLight : public Light {
	public:
		AreaLight() {}
		AreaLight(hitable* obj)
		{
			m_object = obj;
		}

		virtual ~AreaLight() {}

	public:
		virtual real getPdf(const vec3& org, sampler* sampler) const override final
		{
			real pdf = 0;

			if (m_object) {
				hitrecord rec;

				auto pos = m_object->getRandomPosOn(sampler);

				ray r(
					org,
					normalize(pos - org));

				bool isHit = m_object->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					pdf = 1 / rec.area;
				}
			}

			return pdf;
		}

		virtual vec3 sampleDirToLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 dir;

			if (m_object) {
				hitrecord rec;

				auto pos = m_object->getRandomPosOn(sampler);

				ray r(
					org,
					normalize(pos - org));

				bool isHit = m_object->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					dir = rec.p - r.org;
				}
			}

			return std::move(dir);
		}

		virtual vec3 sampleNormalOnLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 nml;

			if (m_object) {
				hitrecord rec;

				auto pos = m_object->getRandomPosOn(sampler);

				ray r(
					org,
					normalize(pos - org));

				bool isHit = m_object->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					nml = rec.normal;
				}
			}

			return std::move(nml);
		}

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			if (m_object) {
				hitrecord rec;

				auto pos = m_object->getRandomPosOn(sampler);

				ray r(
					org,
					normalize(pos - org));

				bool isHit = m_object->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					result.pdf = 1 / rec.area;
					result.dir = rec.p - r.org;
					result.nml = rec.normal;
					result.le = m_le;
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

	private:
		hitable* m_object{ nullptr };
	};
}