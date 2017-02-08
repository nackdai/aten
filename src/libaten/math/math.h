#pragma once

#include <math.h>
#include "types.h"

#define AT_MATH_PI	CONST_REAL(3.14159265358979323846)

#ifdef TYPE_DOUBLE
	#define AT_MATH_INF         (1e64)
	#define AT_MATH_EPSILON     (1e-6)
#else
	#define AT_MATH_INF         (1e32)
	#define AT_MATH_EPSILON     (1e-6)
#endif

namespace aten {
	inline real sqrt(real f)
	{
#ifdef TYPE_DOUBLE
		return ::sqrt(f);
#else
		return ::sqrtf(f);
#endif
	}
}