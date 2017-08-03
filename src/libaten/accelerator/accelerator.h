#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	struct BVHNode {
		float hit{ -1 };		///< Link index if ray hit.
		float miss{ -1 };		///< Link index if ray miss.
		float parent{ -1 };		///< Parent node index.
		float padding0;

		float shapeid{ -1 };	///< Object index.
		float primid{ -1 };		///< Triangle index.
		float exid{ -1 };		///< External bvh index.
		float meshid{ -1 };		///< Mesh id.

		aten::vec4 boxmin;		///< AABB min position.
		aten::vec4 boxmax;		///< AABB max position.

		bool isLeaf() const
		{
			return (shapeid >= 0 || primid >= 0);
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
