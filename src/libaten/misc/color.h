#pragma once

#include "types.h"

namespace aten {
	template <typename _T>
	struct TColor {
		union {
			struct {
				_T r;
				_T g;
				_T b;
				_T a;
			};
			_T c[4];
		};
	};

	using color = TColor<uint8_t>;
}