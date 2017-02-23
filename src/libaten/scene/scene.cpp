#include "scene/scene.h"

namespace aten {
	bool scene::hitLight(
		const Light* light,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		bool isHit = hit(r, t_min, t_max, rec);

		auto lightobj = light->getLightObject();

		if (lightobj) {
			if (isHit && rec.obj == lightobj) {
				return true;
			}
		}

		if (light->isSingular()) {
			if (!isHit) {
				return true;
			}
		}

		return false;
	}
}
