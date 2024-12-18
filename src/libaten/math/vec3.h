#pragma once

#ifdef __NVCC__
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#endif

#include "glm/glm.hpp"

#include "defs.h"
#include "math/math.h"
#include "misc/tuple.h"

namespace aten {
#if 0
    class vec3 {
    public:
        union {
            struct {
                float x, y, z;
            };
            struct {
                float r, g, b;
            };
            float a[3];
        };

        vec3()
        {
            x = y = z = float(0);
        }

        vec3(float _x, float _y, float _z)
        {
            x = _x;
            y = _y;
            z = _z;
        }

        vec3(float f)
        {
            x = y = z = f;
        }

        inline AT_HOST_DEVICE_API const vec3& operator+() const
        {
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3 operator-() const
        {
            vec3 ret(0);
            ret.x = -x;
            ret.y = -y;
            ret.z = -z;
            return ret;
        }
        inline AT_HOST_DEVICE_API float operator[](int32_t i) const
        {
            return a[i];
        }
        inline AT_HOST_DEVICE_API float& operator[](int32_t i)
        {
            return a[i];
        };

        inline AT_HOST_DEVICE_API vec3& operator+=(const vec3& v)
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3& operator-=(const vec3& v)
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3& operator*=(const vec3& v)
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3& operator/=(const vec3& v)
        {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3& operator*=(const float t)
        {
            x *= t;
            y *= t;
            z *= t;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec3& operator/=(const float t)
        {
            x /= t;
            y /= t;
            z /= t;
            return *this;
        }
    };

    inline AT_HOST_DEVICE_API vec3 operator+(const vec3& v1, const vec3& v2)
    {
        vec3 ret = aten::vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator+(const vec3& v1, float f)
    {
        vec3 ret = aten::vec3(v1.x + f, v1.y + f, v1.z + f);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator-(const vec3& v1, const vec3& v2)
    {
        vec3 ret = aten::vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator-(const vec3& v1, float f)
    {
        vec3 ret = aten::vec3(v1.x - f, v1.y - f, v1.z - f);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator*(const vec3& v1, const vec3& v2)
    {
        vec3 ret = aten::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator/(const vec3& v1, const vec3& v2)
    {
        vec3 ret = aten::vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator*(float t, const vec3& v)
    {
        vec3 ret = aten::vec3(t * v.x, t * v.y, t * v.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator*(const vec3& v, float t)
    {
        vec3 ret = aten::vec3(t * v.x, t * v.y, t * v.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator/(const vec3& v, float t)
    {
        vec3 ret = aten::vec3(v.x / t, v.y / t, v.z / t);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator/(float t, const vec3& v)
    {
        vec3 ret = aten::vec3(t / v.x, t / v.y, t / v.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API float dot(const vec3& v1, const vec3& v2)
    {
        auto ret = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 cross(const vec3& v1, const vec3& v2)
    {
        vec3 ret = aten::vec3(
            v1.a[1] * v2.a[2] - v1.a[2] * v2.a[1],
            v1.a[2] * v2.a[0] - v1.a[0] * v2.a[2],
            v1.a[0] * v2.a[1] - v1.a[1] * v2.a[0]);

        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 normalize(const vec3& v)
    {
        auto invLen = aten::rsqrt(dot(v, v));
        auto ret = v * invLen;
        return ret;
    }

    inline AT_HOST_DEVICE_API float length(const vec3& v)
    {
        auto ret = aten::sqrt(dot(v, v));
        return ret;
    }
#else
#ifdef TYPE_DOUBLE
    using vec3 = glm::dvec3;

    inline AT_HOST_DEVICE_API vec3 operator*(float t, const vec3& v)
    {
        vec3 ret(t * v.x, t * v.y, t * v.z);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 operator*(const vec3& v, float t)
    {
        vec3 ret(t * v.x, t * v.y, t * v.z);
        return ret;
    }
#else
    using vec3 = glm::highp_vec3;
#endif
#endif

    inline AT_HOST_DEVICE_API void set(vec3& v, float x, float y, float z)
    {
        v.x = x;
        v.y = y;
        v.z = z;
    }

    inline AT_HOST_DEVICE_API float squared_length(const vec3& v)
    {
        auto ret = dot(v, v);
        return ret;
    }

    inline AT_HOST_DEVICE_API float length(const vec3& v)
    {
        auto ret = aten::sqrt(dot(v, v));
        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 vmin(const vec3& a, const vec3& b)
    {
        return vec3(
            aten::min(a.x, b.x),
            aten::min(a.y, b.y),
            aten::min(a.z, b.z));
    }

    inline AT_HOST_DEVICE_API vec3 vmax(const vec3& a, const vec3& b)
    {
        return vec3(
            aten::max(a.x, b.x),
            aten::max(a.y, b.y),
            aten::max(a.z, b.z));
    }

    inline AT_HOST_DEVICE_API vec3 pow(const vec3& v, float a)
    {
        vec3 ret(
            pow(v.x, a),
            pow(v.y, a),
            pow(v.z, a));

        return ret;
    }

    inline AT_HOST_DEVICE_API vec3 mix(const vec3& v0, const vec3& v1, float a)
    {
        vec3 ret = v0 * (float(1) - a) + v1 * a;
        return ret;
    }

    union _vec3_cmp_res {
        struct {
            uint8_t _0 : 1;
            uint8_t _1 : 1;
            uint8_t _2 : 1;
            uint8_t padding : 5;
        };
        uint8_t f;
    };

    inline AT_HOST_DEVICE_API int32_t cmpGEQ(const vec3& a, const vec3& b)
    {
        _vec3_cmp_res res;

        res.f = 0;
        res._0 = (a.x >= b.x);
        res._1 = (a.y >= b.y);
        res._2 = (a.z >= b.z);

        return res.f;
    }

    static inline AT_HOST_DEVICE_API aten::vec3 GetOrthoVector(const aten::vec3& n)
    {
        aten::vec3 p;

        // NOTE
        // dotを計算したときにゼロになるようなベクトル.
        // k は normalize 計算用.

#if 0
        if (aten::abs(n.z) > float(0.707106781186547524401)) {
            float k = aten::sqrt(n.y * n.y + n.z * n.z);
            p.x = 0;
            p.y = -n.z / k;
            p.z = n.y / k;
        }
        else {
            float k = aten::sqrt(n.x * n.x + n.y * n.y);
            p.x = -n.y / k;
            p.y = n.x / k;
            p.z = 0;
        }
#else
        if (aten::abs(n.z) > float(0)) {
            float k = aten::sqrt(n.y * n.y + n.z * n.z);
            p.x = 0;
            p.y = -n.z / k;
            p.z = n.y / k;
        }
        else {
            float k = aten::sqrt(n.x * n.x + n.y * n.y);
            p.x = n.y / k;
            p.y = -n.x / k;
            p.z = 0;
        }

        p = normalize(p);
#endif

        return p;
    }

    static inline AT_HOST_DEVICE_API aten::tuple<aten::vec3, aten::vec3> GetTangentCoordinate(const aten::vec3& n)
    {
        auto t = GetOrthoVector(n);
        auto b = cross(n, t);
        t = cross(b, n);
        return aten::make_tuple(t, b);
    }

    inline AT_HOST_DEVICE_API bool isInvalid(const vec3& v)
    {
        return isInvalid(v.x) || isInvalid(v.y) || isInvalid(v.z);
    }

    template <class T>
    inline AT_HOST_DEVICE_API float max_from_vec3(const T& v)
    {
        return std::max(std::max(v.x, v.y), v.z);
    }

#ifdef __CUDACC__
    template <>
    inline AT_HOST_DEVICE_API float max_from_vec3(const float3& v)
    {
        return max(max(v.x, v.y), v.z);
    }
#endif

    template <class T>
    inline AT_HOST_DEVICE_API float min_from_vec3(const T& v)
    {
        return std::min(std::min(v.x, v.y), v.z);
    }

#ifdef __CUDACC__
    template <>
    inline AT_HOST_DEVICE_API float min_from_vec3(const float3& v)
    {
        return min(min(v.x, v.y), v.z);
    }
#endif

#ifdef __CUDACC__
    inline AT_HOST_DEVICE_API vec3 operator*(const vec3& a, const float3& b)
    {
        return aten::vec3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    inline AT_HOST_DEVICE_API vec3 operator*(const float3& a, const vec3& b)
    {
        return aten::vec3(a.x * b.x, a.y * b.y, a.z * b.z);
    }
#endif
}

#ifdef __NVCC__
#pragma nv_diag_default = esa_on_defaulted_function_ignored
#endif
