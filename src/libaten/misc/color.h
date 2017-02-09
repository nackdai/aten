#pragma once

#include "types.h"

namespace aten {
	struct color {
		union {
			struct {
				uint8_t r;
				uint8_t g;
				uint8_t b;
				uint8_t a;
			};
			uint8_t c[4];
		};
	};
}