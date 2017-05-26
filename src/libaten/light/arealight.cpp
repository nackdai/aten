#include "light/arealight.h"

namespace AT_NAME {
	aten::LightSampleResult AreaLight::sample(const aten::vec3& org, aten::sampler* sampler) const
	{
		bool isHit = false;
		const aten::hitable* obj = (aten::hitable*)m_param.object.ptr;

		aten::LightSampleResult result;

		if (obj) {
			aten::ray r;
			aten::hitrecord rec;
			aten::hitrecordOption recOpt;

			if (sampler) {
				aten::hitable::SamplePosNormalPdfResult result;
				result.idx[0] = -1;

				obj->getSamplePosNormalArea(&result, sampler);

				auto pos = result.pos;
				auto dir = pos - org;
				r = aten::ray(org, dir);

				if (result.idx[0] >= 0) {
					rec.t = dir.length();

					recOpt.idx[0] = result.idx[0];
					recOpt.idx[1] = result.idx[1];
					recOpt.idx[2] = result.idx[2];

					recOpt.a = result.a;
					recOpt.b = result.b;
				}
				else {
					obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec, recOpt);
				}
			}
			else {
				auto pos = obj->getBoundingbox().getCenter();

				auto dir = pos - org;
				r = aten::ray(org, dir);

				obj->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec, recOpt);
			}

			aten::hitable::evalHitResult(obj, r, rec, recOpt);

			sample(
				&result,
				&rec,
				&this->param(),
				org,
				sampler);
		}

		return std::move(result);
	}
}
