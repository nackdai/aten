#pragma once

#include <math.h>
#include <algorithm>
#include "types.h"
#include "defs.h"

#define AT_MATH_PI		real(3.14159265358979323846)
#define AT_MATH_PI_2	real(AT_MATH_PI * 2)
#define AT_MATH_PI_HALF	real(AT_MATH_PI * 0.5)

#if 1
#ifdef TYPE_DOUBLE
	#define AT_MATH_INF				DBL_MAX
	//#define AT_MATH_EPSILON		DBL_EPSILON
	#define AT_MATH_EPSILON			(1e-6)
#else
	#define AT_MATH_INF				FLT_MAX
	//#define AT_MATH_EPSILON		FLT_EPSILON
	#define AT_MATH_EPSILON			(float)(1e-3)
#endif
#else
#ifdef TYPE_DOUBLE
	#define AT_MATH_INF				(1e64)
	#define AT_MATH_EPSILON			(1e-6)
	#define AT_MATH_EPSILIN_SQRT	(1e-3)
#else
	#define AT_MATH_INF				(float)(1e32)
	#define AT_MATH_EPSILON			(float)(1e-3)
#endif
#endif

#define Deg2Rad(d)   (AT_MATH_PI * (d) / real(180.0))
#define Rad2Deg(r)   ((r) * real(180.0) / AT_MATH_PI)

#ifdef TYPE_DOUBLE
	#define AT_MATH_FUNC(func, v)	func(v)
	#define AT_MATH_FUNC2(func, v0, v1)	func(v0, v1)
#else
	#define AT_MATH_FUNC(func, v)	func##f(v)
	#define AT_MATH_FUNC2(func, v0, v1)	func##f(v0, v1)
#endif

namespace aten {
	inline AT_DEVICE_API real sqrt(real f)
	{
		return AT_MATH_FUNC(::sqrt, f);
	}

	inline AT_DEVICE_API real rsqrt(real f)
	{
#ifdef __CUDACC__
		return rsqrtf(f);
#else
		return real(1) / aten::sqrt(f);
#endif
	}

	inline real tan(real f)
	{
		return AT_MATH_FUNC(::tan, f);
	}

	inline AT_DEVICE_API real cos(real f)
	{
		return AT_MATH_FUNC(::cos, f);
	}

	inline AT_DEVICE_API real sin(real f)
	{
		return AT_MATH_FUNC(::sin, f);
	}

	inline real atan2(real y, real x)
	{
		return AT_MATH_FUNC2(::atan2, y, x);
	}

	inline AT_DEVICE_API real atan(real f)
	{
		return AT_MATH_FUNC(::atan, f);
	}

	inline AT_DEVICE_API real asin(real f)
	{
		return AT_MATH_FUNC(::asin, f);
	}

	inline real acos(real f)
	{
		return AT_MATH_FUNC(::acos, f);
	}

	inline AT_DEVICE_API real log(real f)
	{
		return AT_MATH_FUNC(::log, f);
	}

	inline AT_DEVICE_API real exp(real f)
	{
		return AT_MATH_FUNC(::exp, f);
	}

	inline AT_DEVICE_API real pow(real f, real v)
	{
		return AT_MATH_FUNC2(::pow, f, v);
	}

	inline AT_DEVICE_API real abs(real f)
	{
#ifdef TYPE_DOUBLE
		return ::abs(f);
#else
		return ::fabsf(f);
#endif
	}

	inline real floor(real f)
	{
		return AT_MATH_FUNC(::floor, f);
	}

	inline real ceil(real f)
	{
		return AT_MATH_FUNC(::ceil, f);
	}

#ifdef __CUDACC__
	template <typename T>
	inline AT_DEVICE_API T cmpMax(T a, T b)
	{
		return (a > b ? a : b);
	}

	template <>
	inline AT_DEVICE_API float cmpMax(float a, float b)
	{
		return fmaxf(a, b);
	}

	template <>
	inline AT_DEVICE_API int cmpMax(int a, int b)
	{
		return max(a, b);
	}
#else
	template <typename T>
	inline AT_DEVICE_API T cmpMax(T a, T b)
	{
		return std::max<T>(a, b);
	}
#endif

#ifdef __CUDACC__
	template <typename T>
	inline AT_DEVICE_API T cmpMin(T a, T b)
	{
		return (a < b ? a : b);
	}

	template <>
	inline AT_DEVICE_API float cmpMin(float a, float b)
	{
		return fminf(a, b);
	}

	template <>
	inline AT_DEVICE_API int cmpMin(int a, int b)
	{
		return min(a, b);
	}
#else
	template <typename T>
	inline AT_DEVICE_API T cmpMin(T a, T b)
	{
		return std::min<T>(a, b);
	}
#endif


	template <typename _T>
	inline AT_DEVICE_API _T clamp(_T f, _T a, _T b)
	{
		return cmpMin(cmpMax(f, a), b);
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

	// Neary Equal.
#if 0
	// https://github.com/scijs/almost-equal/blob/master/almost_equal.js
	inline bool isClose(real a, real b, real relativeError = AT_MATH_EPSILON, real absoluteError = AT_MATH_EPSILON)
	{
		auto d = abs(a - b);


		if (d <= absoluteError) {
			return true;
		}

		if (d <= relativeError * std::min(abs(a), abs(b))) {
			return true;
		}

		return false;
	}
#else
	// http://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
	inline bool isClose(real a, real b, real relativeError = AT_MATH_EPSILON, real absoluteError = real(0))
	{
		auto d = abs(a - b);
		bool result = d <= std::max(relativeError * std::max(abs(a), abs(b)), absoluteError);
		return result;
	}
#endif
}