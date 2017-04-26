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

		bool AT_DEVICE_API isLeaf() const
		{
			return (left <= 0 && right <= 0);
		}
	};

	class accelerator : public hitable {
	public:
		accelerator() {}
		virtual ~accelerator() {}

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
