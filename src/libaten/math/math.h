#pragma once

#include <math.h>
#include <algorithm>
#include "types.h"
#include "defs.h"

#define AT_MATH_PI        real(3.14159265358979323846)
#define AT_MATH_PI_2    real(AT_MATH_PI * 2)
#define AT_MATH_PI_HALF    real(AT_MATH_PI * 0.5)

#if 1
#ifdef TYPE_DOUBLE
    #define AT_MATH_INF                DBL_MAX
    //#define AT_MATH_EPSILON        DBL_EPSILON
    #define AT_MATH_EPSILON            (1e-6)
#else
    #define AT_MATH_INF                FLT_MAX
    //#define AT_MATH_EPSILON        FLT_EPSILON
    #define AT_MATH_EPSILON            (float)(1e-3)
#endif
#else
#ifdef TYPE_DOUBLE
    #define AT_MATH_INF                (1e64)
    #define AT_MATH_EPSILON            (1e-6)
    #define AT_MATH_EPSILIN_SQRT    (1e-3)
#else
    #define AT_MATH_INF                (float)(1e32)
    #define AT_MATH_EPSILON            (float)(1e-3)
#endif
#endif

#define Deg2Rad(d)   (AT_MATH_PI * (d) / real(180.0))
#define Rad2Deg(r)   ((r) * real(180.0) / AT_MATH_PI)

#ifdef TYPE_DOUBLE
    #define AT_MATH_FUNC(func, v)    func(v)
    #define AT_MATH_FUNC2(func, v0, v1)    func(v0, v1)
#else
    #define AT_MATH_FUNC(func, v)    func##f(v)
    #define AT_MATH_FUNC2(func, v0, v1)    func##f(v0, v1)
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

    inline AT_DEVICE_API real tan(real f)
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

    inline AT_DEVICE_API real atan2(real y, real x)
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

    inline AT_DEVICE_API real acos(real f)
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

    template <typename T>
    inline AT_DEVICE_API T abs(T f)
    {
        return ::abs(f);
    }

    template <>
    inline AT_DEVICE_API real abs(real f)
    {
#ifdef TYPE_DOUBLE
        return ::abs(f);
#else
        return ::fabsf(f);
#endif
    }

    inline AT_DEVICE_API real floor(real f)
    {
        return AT_MATH_FUNC(::floor, f);
    }

    inline AT_DEVICE_API real ceil(real f)
    {
        return AT_MATH_FUNC(::ceil, f);
    }

    template <typename T>
    inline AT_DEVICE_API void swapVal(T& a, T& b)
    {
        T tmp = a;
        a = b;
        b = tmp;
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
        bool b = std::isnan(f);
        if (!b) {
            b = std::isinf(f);
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

    inline uint32_t nextPow2(uint32_t n)
    {
        // NOTE
        // I think algortihm in http://aggregate.org/MAGIC/#Next%20Largest%20Power%20of%202 is not correct.
        // I found that 1 is subtracted from argument value in WWW.

        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;

        return n + 1;
    }

    template <typename TYPE>
    inline AT_DEVICE_API TYPE mix(const TYPE& x, const TYPE& y, real a)
    {
        // Linear interpolation.
        // x(1-a)+y*a
        return x * (1 - a) + y * a;
    }

    // Neary Equal.
    inline AT_DEVICE_API bool isClose(real A, real B, int maxUlps)
    {
#ifdef TYPE_DOUBLE
        // TODO
        return aten::abs(A - B) < AT_MATH_EPSILON;
#else
        // NOTE
        // http://www.cygnus-software.com/papers/comparingfloats/Comparing%20floating%20point%20numbers.htm

        // Make sure maxUlps is non-negative and small enough that the
        // default NAN won't compare as equal to anything.
        AT_ASSERT(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
        
        // Make aInt lexicographically ordered as a twos-complement int
        int aInt = *(int*)&A;
        if (aInt < 0) {
            aInt = 0x80000000 - aInt;
        }

        // Make bInt lexicographically ordered as a twos-complement int
        int bInt = *(int*)&B;
        if (bInt < 0) {
            bInt = 0x80000000 - bInt;
        }

        int intDiff = abs(aInt - bInt);
        if (intDiff <= maxUlps) {
            return true;
        }

        return false;
#endif
    }

    // 31bit float (not have sign bit) expand to correct float (32 bit float, have sign bit).
    // Only return positive value.
    inline AT_DEVICE_API float expandTo32bitFloat(uint32_t n)
    {
        union {
            float f;
            struct {
                uint32_t n : 31;
                uint32_t sign : 1;
            };
        } val;

        val.n = n;
        val.sign = 0;

        return val.f;
    }

    // Collapse 32 bit float to 31 bit float (not have sign bit) as integer.
    // But miss sign bit.
    inline AT_DEVICE_API uint32_t collapseTo31bitInteger(float f)
    {
        union {
            float f;
            struct {
                uint32_t n : 31;
                uint32_t sign : 1;
            };
        } val;

        val.f = f;

        return val.n;
    }
}
