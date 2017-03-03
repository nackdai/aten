#pragma once

#include <math.h>
#include <algorithm>
#include "types.h"

#define AT_MATH_PI		aten::real(3.14159265358979323846)
#define AT_MATH_PI_2	aten::real(AT_MATH_PI * 2)
#define AT_MATH_PI_HALF	aten::real(AT_MATH_PI * 0.5)

#ifdef TYPE_DOUBLE
	#define AT_MATH_INF         (1e64)
	#define AT_MATH_EPSILON     (1e-6)
#else
	#define AT_MATH_INF         (float)(1e32)
	#define AT_MATH_EPSILON     (float)(1e-6)
#endif

#define Deg2Rad(d)   (AT_MATH_PI * (d) / aten::real(180.0))
#define Rad2Deg(r)   ((r) * aten::real(180.0) / AT_MATH_PI)

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

	inline real atan(real f)
	{
#ifdef TYPE_DOUBLE
		return ::atan(f);
#else
		return ::atanf(f);
#endif
	}

	inline real asin(real f)
	{
#ifdef TYPE_DOUBLE
		return ::asin(f);
#else
		return ::asinf(f);
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

	inline real abs(real f)
	{
#ifdef TYPE_DOUBLE
		return (f < real(0) ? -f : f);
#else
		return ::fabs(f);
#endif
	}

	template <typename _T>
	inline _T clamp(_T f, _T a, _T b)
	{
		return std::min(std::max(f, a), b);
	}

	inline bool isValid(real f)
	{
		bool b = isnan(f);
		if (!b) {
			b = isinf(f);
		}

		return !b;
	}

	inline bool isInvalid(real f)
	{
		bool b = !isValid(f);
		return b;
	}

	inline int clz(uint32_t x)
	{
		// NOTE
		// NLZ
		// http://www.nminoru.jp/~nminoru/programming/bitcount.html

		int y, m, n;

		y = -(int)(x >> 16);
		m = (y >> 16) & 16;
		n = 16 - m;
		x = x >> m;

		y = x - 0x100;
		m = (y >> 16) & 8;
		n = n + m;
		x = x << m;

		y = x - 0x1000;
		m = (y >> 16) & 4;
		n = n + m;
		x = x << m;

		y = x - 0x4000;
		m = (y >> 16) & 2;
		n = n + m;
		x = x << m;

		y = x >> 14;
		m = y & ~(y >> 1);

		return n + 2 - m;
	}
}