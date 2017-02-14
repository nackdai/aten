#pragma once

#include <math.h>
#include "types.h"

#define AT_MATH_PI	CONST_REAL(3.14159265358979323846)

#ifdef TYPE_DOUBLE
	#define AT_MATH_INF         (1e64)
	#define AT_MATH_EPSILON     (1e-6)
#else
	#define AT_MATH_INF         (float)(1e32)
	#define AT_MATH_EPSILON     (float)(1e-6)
#endif

#define Deg2Rad(d)   (AT_MATH_PI * (d) / CONST_REAL(180.0))
#define Rad2Deg(r)   ((r) * CONST_REAL(180.0) / AT_MATH_PI)

namespace aten {
	inline real sqrt(real f)
	{
#ifdef TYPE_DOUBLE
		return ::sqrt(f);
#else
		return ::sqrtf(f);
#endif
	}

	inline real tan(real f)
	{
#ifdef TYPE_DOUBLE
		return ::tan(f);
#else
		return ::tanf(f);
#endif
	}

	inline real cos(real f)
	{
#ifdef TYPE_DOUBLE
		return ::cos(f);
#else
		return ::cosf(f);
#endif
	}

	inline real sin(real f)
	{
#ifdef TYPE_DOUBLE
		return ::sin(f);
#else
		return ::sinf(f);
#endif
	}

	inline real atan2(real y, real x)
	{
#ifdef TYPE_DOUBLE
		return ::atan2(y, x);
#else
		return ::atan2f(y, x);
#endif
	}

	inline real acos(real f)
	{
#ifdef TYPE_DOUBLE
		return ::acos(f);
#else
		return ::acosf(f);
#endif
	}

	inline real log(real f)
	{
#ifdef TYPE_DOUBLE
		return ::log(f);
#else
		return ::logf(f);
#endif
	}

	inline real exp(real f)
	{
#ifdef TYPE_DOUBLE
		return ::exp(f);
#else
		return ::expf(f);
#endif
	}

	inline real pow(real f, real v)
	{
#ifdef TYPE_DOUBLE
		return ::pow(f, v);
#else
		return ::powf(f, v);
#endif
	}

	template <typename _T>
	inline _T clamp(_T f, _T a, _T b)
	{
		return min(max(f, a), b);
	}
}