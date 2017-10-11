#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	class accelerator : public hitable {
	public:
		accelerator() {}
		virtual ~accelerator() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) = 0;
	};
}
