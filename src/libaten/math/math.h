#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

#include "types.h"
#include "defs.h"

constexpr auto AT_MATH_PI = 3.14159265358979323846F;
constexpr auto AT_MATH_PI_2 = AT_MATH_PI * 2;
constexpr auto AT_MATH_PI_HALF = AT_MATH_PI * 0.5F;

constexpr auto AT_MATH_INF = std::numeric_limits<float>::max();
constexpr auto AT_MATH_EPSILON = 1e-3F;

namespace aten {
    constexpr inline AT_HOST_DEVICE_API float Deg2Rad(float d)
    {
        return (AT_MATH_PI * (d) / 180.0F);
    }

    constexpr inline AT_HOST_DEVICE_API float Rad2Deg(float r)
    {
        return ((r) * 180.0F / AT_MATH_PI);
    }

    inline AT_HOST_DEVICE_API float sqrt(float f)
    {
        return std::sqrt(f);
    }

    inline AT_HOST_DEVICE_API float rsqrt(float f)
    {
#ifdef __CUDACC__
        return rsqrtf(f);
#else
        return 1.0F / aten::sqrt(f);
#endif
    }

    inline AT_HOST_DEVICE_API float sqr(float f)
    {
        return f * f;
    }

    inline AT_HOST_DEVICE_API float tan(float f)
    {
        return std::tan(f);
    }

    inline AT_HOST_DEVICE_API float cos(float f)
    {
        return std::cos(f);
    }

    inline AT_HOST_DEVICE_API float sin(float f)
    {
        return std::sin(f);
    }

    inline AT_HOST_DEVICE_API float sign(float f)
    {
        if (f == 0.0F) {
            return 0.0F;
        }
        else if (f > 0.0F) {
            return 1.0F;
        }
        else {
            return -1.0F;
        }
    }

    inline AT_HOST_DEVICE_API float atan2(float y, float x)
    {
        return std::atan2(y, x);
    }

    inline AT_HOST_DEVICE_API float atan(float f)
    {
        return std::atan(f);
    }

    inline AT_HOST_DEVICE_API float asin(float f)
    {
        return std::asin(f);
    }

    inline AT_HOST_DEVICE_API float acos(float f)
    {
        return std::acos(f);
    }

    inline AT_HOST_DEVICE_API float log(float f)
    {
        return std::log(f);
    }

    inline AT_HOST_DEVICE_API float exp(float f)
    {
        return std::exp(f);
    }

    inline AT_HOST_DEVICE_API float pow(float f, float v)
    {
        return std::pow(f, v);
    }

    template <class T>
    inline AT_HOST_DEVICE_API T abs(T f)
    {
        return std::abs(f);
    }

    inline AT_HOST_DEVICE_API float floor(float f)
    {
        return std::floor(f);
    }

    inline AT_HOST_DEVICE_API float ceil(float f)
    {
        return std::ceil(f);
    }

#ifdef __CUDACC__
    template <class T>
    inline AT_HOST_DEVICE_API auto max(T a, T b) -> std::enable_if_t<std::is_fundamental_v<T>, T>
    {
        return (a > b ? a : b);
    }

    template <>
    inline AT_HOST_DEVICE_API float max(float a, float b)
    {
        return fmaxf(a, b);
    }

    template <>
    inline AT_HOST_DEVICE_API int32_t max(int32_t a, int32_t b)
    {
        return ::max(a, b);
    }
#else
    template <class T>
    inline AT_HOST_DEVICE_API auto max(T a, T b) -> std::enable_if_t<std::is_fundamental_v<T>, T>
    {
        return std::max<T>(a, b);
    }
#endif

#ifdef __CUDACC__
    template <class T>
    inline AT_HOST_DEVICE_API auto min(T a, T b) -> std::enable_if_t<std::is_fundamental_v<T>, T>
    {
        return (a < b ? a : b);
    }

    template <>
    inline AT_HOST_DEVICE_API float min(float a, float b)
    {
        return fminf(a, b);
    }

    template <>
    inline AT_HOST_DEVICE_API int32_t min(int32_t a, int32_t b)
    {
        return ::min(a, b);
    }
#else
    template <class T>
    inline AT_HOST_DEVICE_API auto min(T a, T b) -> std::enable_if_t<std::is_fundamental_v<T>, T>
    {
        return std::min<T>(a, b);
    }
#endif

    template <class T>
    inline AT_HOST_DEVICE_API T clamp(T f, T a, T b)
    {
        return cmpMin(cmpMax(f, a), b);
    }

    template <class T>
    inline AT_HOST_DEVICE_API auto saturate(T f) -> std::enable_if_t<std::is_floating_point_v<T>, T>
    {
        return clamp(f, T(0), T(1));
    }

    inline AT_HOST_DEVICE_API bool isInvalid(float f)
    {
#ifdef __CUDACC__
        bool is_invalid = isnan(f) || isinf(f);
#else
        bool is_invalid = std::isnan(f) || std::isinf(f);
#endif
        return is_invalid;
    }

    inline AT_HOST_DEVICE_API bool isValid(float f)
    {
        return !isInvalid(f);
    }

    inline int32_t clz(uint32_t x)
    {
        // NOTE
        // NLZ
        // http://www.nminoru.jp/~nminoru/programming/bitcount.html

        int32_t y, m, n;

        y = -(int32_t)(x >> 16);
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

#ifdef TYPE_DOUBLE
    inline AT_HOST_DEVICE_API int64_t floatAsInt(double f)
    {
        // NOTE
        // Accroding C# implementaion
        // DoubleToInt64Bits
        // https://referencesource.microsoft.com/#mscorlib/system/bitconverter.cs,db20ea77a561c0ac,references
        // It is possible to convert like below
        return *(int64_t*)&f;
    }

    inline AT_HOST_DEVICE_API double intAsFloat(int64_t n)
    {
        // NOTE
        // Accroding C# implementaion
        // Int64BitsToDouble
        // https://referencesource.microsoft.com/#mscorlib/system/bitconverter.cs,db20ea77a561c0ac,references
        // It is possible to convert like below
        return *(double*)&n;
    }
#else
    inline AT_HOST_DEVICE_API int32_t floatAsInt(float f)
    {
        return *(int32_t*)&f;
    }

    inline AT_HOST_DEVICE_API float intAsFloat(int32_t n)
    {
        return *(float*)&n;
    }
#endif

    template <class T>
    inline AT_HOST_DEVICE_API T mix(const T& x, const T& y, float a)
    {
        // Linear interpolation.
        // x(1-a)+y*a
        return x * (1 - a) + y * a;
    }

    // Neary Equal.
    inline AT_HOST_DEVICE_API bool isClose(float A, float B, int32_t maxUlps)
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

        // Make aInt lexicographically ordered as a twos-complement int32_t
        int32_t aInt = *(int32_t*)&A;
        if (aInt < 0) {
            aInt = 0x80000000 - aInt;
        }

        // Make bInt lexicographically ordered as a twos-complement int32_t
        int32_t bInt = *(int32_t*)&B;
        if (bInt < 0) {
            bInt = 0x80000000 - bInt;
        }

        int32_t intDiff = aten::abs<int32_t>(aInt - bInt);
        if (intDiff <= maxUlps) {
            return true;
        }

        return false;
#endif
    }


    inline AT_HOST_DEVICE_API bool isfinite(float f)
    {
#ifdef __CUDACC__
        return isfinite(f);
#else
        return std::isfinite(f);
#endif
    }

    template <class T>
    constexpr inline AT_HOST_DEVICE_API bool IsInRange(T x, T a, T b)
    {
        return a <= x && x <= b;
    }
}
