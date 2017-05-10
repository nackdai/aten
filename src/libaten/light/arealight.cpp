#include "light/arealight.h"

namespace AT_NAME {
	aten::LightSampleResult AreaLight::sample(const aten::vec3& org, aten::sampler* sampler) const
	{
		auto funcHitTest = [](const aten::vec3& o, const aten::UnionIdxPtr& object, aten::vec3& pos, aten::sampler* smpl, aten::hitrecord& rec)
		{
			bool isHit = false;
			const aten::hitable* obj = (aten::hitable*)object.ptr;

			if (obj) {
				if (smpl) {
					pos = obj->getRandomPosOn(smpl);
				}
				else {
					pos = obj->getBoundingbox().getCenter();
				}

				auto dir = pos - o;
				auto dist = dir.length();

				aten::ray r(o, aten::normalize(dir));

				isHit = obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);
			}

			return isHit;
		};

		aten::LightSampleResult result;

		sample(
			funcHitTest,
			&result,
			&this->param(),
			org,
			sampler);

		return std::move(result);
	}
}
