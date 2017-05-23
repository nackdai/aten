#include "light/arealight.h"

namespace AT_NAME {
	aten::LightSampleResult AreaLight::sample(const aten::vec3& org, aten::sampler* sampler) const
	{
		bool isHit = false;
		const aten::hitable* obj = (aten::hitable*)m_param.object.ptr;

		aten::vec3 pos;
		aten::hitrecord rec;

		aten::LightSampleResult result;

		if (obj) {
			if (sampler) {
				pos = obj->getRandomPosOn(sampler);
			}
			else {
				pos = obj->getBoundingbox().getCenter();
			}

			auto dir = pos - org;
			auto dist = dir.length();

			aten::ray r(org, aten::normalize(dir));

			isHit = obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);
			if (isHit) {
				obj->evalHitResult(r, rec);

				sample(
					&result,
					&rec,
					&this->param(),
					org,
					sampler);
			}
		}

		return std::move(result);
	}
}
