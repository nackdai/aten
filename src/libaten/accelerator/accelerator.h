#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	struct GPUBvhNode {
		float hit{ -1 };		///< Link index if ray hit.
		float miss{ -1 };		///< Link index if ray miss.
		float parent{ -1 };		///< Parent node index.
		float padding0{ 0 };

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

	class bvhnode;

	class accelerator : public hitable {
	public:
		accelerator() {}
		virtual ~accelerator() {}

	public:
		static accelerator* createAccelerator();

		virtual void build(
			hitable** list,
			uint32_t num) = 0;
	};
}
