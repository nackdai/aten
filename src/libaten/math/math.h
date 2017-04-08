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

#ifdef TYPE_DOUBLE
	#define AT_MATH_FUNC(func, v)	func(v)
	#define AT_MATH_FUNC2(func, v0, v1)	func(v0, v1)
#else
	#define AT_MATH_FUNC(func, v)	func##f(f)
	#define AT_MATH_FUNC2(func, v0, v1)	func##f(f0, f1)
#endif

namespace aten {
	inline real sqrt(real f)
	{
		return AT_MATH_FUNC(::sqrt, f);
	}

	inline real tan(real f)
	{
		return AT_MATH_FUNC(::tan, f);
	}

	inline real cos(real f)
	{
		return AT_MATH_FUNC(::cos, f);
	}

	inline real sin(real f)
	{
		return AT_MATH_FUNC(::sin, f);
	}

	inline real atan2(real y, real x)
	{
		return AT_MATH_FUNC2(::atan2, y, x);
	}

	inline real atan(real f)
	{
		return AT_MATH_FUNC(::atan, f);
	}

	inline real asin(real f)
	{
		return AT_MATH_FUNC(::asin, f);
	}

	inline real acos(real f)
	{
		return AT_MATH_FUNC(::acos, f);
	}

	inline real log(real f)
	{
		return AT_MATH_FUNC(::log, f);
	}

	inline real exp(real f)
	{
		return AT_MATH_FUNC(::exp, f);
	}

	inline real pow(real f, real v)
	{
		return AT_MATH_FUNC2(::pow, f, v);
	}

	inline real abs(real f)
	{
		return AT_MATH_FUNC(::abs, f);
	}

	inline real floor(real f)
	{
		return AT_MATH_FUNC(::floor, f);
	}

	inline real ceil(real f)
	{
		return AT_MATH_FUNC(::ceil, f);
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

	template <typename TYPE>
	inline TYPE mix(const TYPE& x, const TYPE& y, real a)
	{
		// Linear interpolation.
		// x(1-a)+y*a
		return x * (1 - a) + y * a;
	}
}