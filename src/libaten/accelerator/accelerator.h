#pragma once

#include <vector>
#include "scene/hitable.h"
#include "math/frustum.h"

namespace aten {
	class accelerator : public hitable {
	public:
		accelerator() {}
		virtual ~accelerator() {}

	public:
		static accelerator* createAccelerator();

		virtual void build(
			hitable** list,
			uint32_t num) = 0;

		struct ResultIntersectTestByFrustum {
			int ep{ -1 };
			int ex{ -1 };
			// TODO

			ResultIntersectTestByFrustum() {}
		};

		virtual ResultIntersectTestByFrustum intersectTestByFrustum(const frustum& f)
		{
			AT_ASSERT(false);
			return std::move(ResultIntersectTestByFrustum());
		}
	};
}
