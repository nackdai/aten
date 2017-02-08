#pragma once

#include <stdint.h>
#include <climits>

namespace aten {
#ifdef TYPE_DOUBLE
	using real = double;
#else
	using real = float;
#endif
}

#define CONST_REAL(v)	v##f