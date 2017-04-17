#include "light/arealight.h"

namespace aten {
	LightSampleResult AreaLight::sample(const vec3& org, sampler* sampler) const
	{
		return std::move(sample(m_param, org, sampler));
	}

	LightSampleResult AreaLight::sample(
		const LightParameter& param,
		const vec3& org,
		sampler* sampler)
	{
		LightSampleResult result;

		const hitable* obj = (hitable*)param.object.ptr;

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
						
					result.le = param.le;
					result.intensity = 1;
					result.finalColor = param.le;

					result.obj = const_cast<hitable*>(obj);

				}
			}
		}

		return std::move(result);
	}

	bool AreaLight::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		return hit(m_param, r, t_min, t_max, rec);
	}

	bool AreaLight::hit(
		const LightParameter& param,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		bool isHit = false;

		const hitable* obj = (hitable*)param.object.ptr;

		if (obj) {
			isHit = obj->hit(r, t_min, t_max, rec);
		}

		return isHit;
	}
}
