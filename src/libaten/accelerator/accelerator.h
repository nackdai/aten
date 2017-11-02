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
			int ep{ -1 };	///< Entry Point.
			int ex{ -1 };	///< Layer Id.

			// 1つ上のレイヤーへの戻り先のノードID.
			int top{ -1 };	///< Upper layer id.

			int padding;

			ResultIntersectTestByFrustum() {}
		};

		virtual bool hitMultiLevel(
			const ResultIntersectTestByFrustum& fisect,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const
		{
			AT_ASSERT(false);
			return false;
		}

		virtual ResultIntersectTestByFrustum intersectTestByFrustum(const frustum& f)
		{
			AT_ASSERT(false);
			return std::move(ResultIntersectTestByFrustum());
		}
	};
}
