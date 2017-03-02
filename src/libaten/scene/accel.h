#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	class accel : public hitable {
	public:
		accel() {}
		virtual ~accel() {}

	public:
		virtual void build(
			bvhnode** list,
			uint32_t num) = 0;
	};
}
