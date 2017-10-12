#pragma once

#include <vector>
#include <tuple>
#include "scene/hitable.h"

namespace aten {
	class bvhnode;

	class accelerator : public hitable {
	public:
		accelerator() {}
		virtual ~accelerator() {}

	public:
		static std::tuple<accelerator*, int> createAccelerator(bool needRegister = false);

		virtual void build(
			hitable** list,
			uint32_t num) = 0;
	};
}
