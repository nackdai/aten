#pragma once

#include <functional>
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

	private:
		template <typename _T>
		_T sample(
			const vec3& org, 
			sampler* sampler, 
			std::function<void(_T&, const vec3&, const hitrecord&)> setter) const
		{
			_T ret = 0;

			if (m_object) {
				hitrecord rec;

				vec3 pos;
				if (sampler) {
					pos = m_object->getRandomPosOn(sampler);
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
						setter(ret, pos, rec);
					}
				}
			}

			return ret;
		}

	public:
		virtual real getPdf(const vec3& org, sampler* sampler) const override final
		{
			real pdf = sample<real>(
				org,
				sampler,
				[](real& result, const vec3& pos, const hitrecord& rec) {
				result = 1 / rec.area;
			});

			return pdf;
		}

		virtual vec3 sampleDirToLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 dir = sample<vec3>(
				org,
				sampler,
				[&](vec3& result, const vec3& pos, const hitrecord& rec) {
				result = rec.p - org;
			});

			return std::move(dir);
		}

		virtual vec3 sampleNormalOnLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 nml = sample<vec3>(
				org,
				sampler,
				[&](vec3& result, const vec3& pos, const hitrecord& rec) {
				result = rec.normal;
			});

			return std::move(nml);
		}

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