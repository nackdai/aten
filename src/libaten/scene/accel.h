#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	struct BVHNode {
		int left{ -1 };
		int right{ -1 };
		int shapeid{ -1 };
		aabb bbox;
	};

	class accel : public hitable {
	public:
		accel() {}
		virtual ~accel() {}

	public:
		virtual void build(
			bvhnode** list,
			uint32_t num) = 0;

		virtual void collectNodes(std::vector<BVHNode>& nodes) const
		{
			// Nothing is done...
		}
	};
}
