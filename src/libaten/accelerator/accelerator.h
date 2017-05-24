#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	struct BVHNode {
		float left{ -1 };
		float right{ -1 };
		float padding[2];

		float shapeid{ -1 };
		float primid{ -1 };
		float nestid{ -1 };
		float exid{ -1 };

		aten::vec4 boxmin;
		aten::vec4 boxmax;

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

		virtual void collectNodes(
			std::vector<std::vector<BVHNode>>& nodes,
			std::vector<aten::mat4>& mtxs) const
		{
			// Nothing is done...
		}
	};
}
