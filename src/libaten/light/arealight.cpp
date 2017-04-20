#include "light/arealight.h"

namespace aten {
	LightSampleResult AreaLight::sample(const vec3& org, sampler* sampler) const
	{
		auto funcHitTest = [](const vec3& o, const UnionIdxPtr& object, vec3& pos, aten::sampler* smpl, hitrecord& rec)
		{
			bool isHit = false;
			const hitable* obj = (hitable*)object.ptr;

			if (obj) {
				if (smpl) {
					pos = obj->getRandomPosOn(smpl);
				}
				else {
					pos = obj->getBoundingbox().getCenter();
				}

				auto dir = pos - o;
				auto dist = dir.length();

				ray r(o, normalize(dir));

				isHit = obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);
			}

			return isHit;
		};

		LightSampleResult result = sample(
			funcHitTest,
			this->param(),
			org,
			sampler);

		return std::move(result);
	}
}
