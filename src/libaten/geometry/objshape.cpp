#include "geometry/objshape.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

#include <iterator>

namespace AT_NAME
{
	void objshape::build()
	{
		aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
		aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

		param.area = 0;

		for (const auto f : faces) {
			f->build(this);
			param.area += f->param.area;

			const auto& faabb = f->getBoundingbox();

			boxmin = aten::min(faabb.minPos(), boxmin);
			boxmax = aten::max(faabb.maxPos(), boxmax);
		}

		m_aabb.init(boxmin, boxmax);
	}
}
