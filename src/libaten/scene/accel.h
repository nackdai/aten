#pragma once

#include <vector>
#include "scene/hitable.h"

namespace aten {
	class accel : public hitable {
	public:
		accel() {}
		virtual ~accel() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) = 0;
	};
}
