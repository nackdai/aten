#pragma once

#include <vector>
#include "scene/hitable.h"

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
	};
}
