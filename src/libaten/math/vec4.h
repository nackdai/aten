#pragma once

#include "defs.h"
#include "math/math.h"
#include "math/vec3.h"
#include "misc/type_traits.h"

namespace aten {
    class vec4 {
    public:
        union {
            vec3 v;
            struct {
                float x, y, z, w;
            };
            struct {
                float r, g, b, a;
            };
            float p[4];
        };

        AT_HOST_DEVICE_API vec4()
        {
            x = y = z = 0;
            w = 1;
        }
        AT_HOST_DEVICE_API vec4(const vec4& _v)
        {
            v = _v.v;
            w = _v.w;
        }
        AT_HOST_DEVICE_API vec4(vec4&& _v) noexcept
        {
            v = _v.v;
            w = _v.w;
        }

        template <
            class V,
            std::enable_if_t<std::is_class_v<V> && !std::is_same_v<V, vec4> && !std::is_same_v<V, vec3> && !aten::is_shared_ptr_v<V>>* = nullptr
        >
        AT_HOST_DEVICE_API vec4(const V& _v)
        {
            x = _v.x;
            y = _v.y;
            z = _v.z;
            w = _v.w;
        }

        AT_HOST_DEVICE_API vec4(float f)
        {
            x = y = z = w = f;
        }
        AT_HOST_DEVICE_API vec4(float _x, float _y, float _z, float _w)
        {
            x = _x;
            y = _y;
            z = _z;
            w = _w;
        }

        template <class T>
        AT_HOST_DEVICE_API vec4(T _x, T _y, T _z)
        {
            x = static_cast<float>(_x);
            y = static_cast<float>(_y);
            z = static_cast<float>(_z);
            w = 1.0F;
        }

        AT_HOST_DEVICE_API vec4(const vec3& _v, float _w)
        {
            v = _v;
            w = _w;
        }
        AT_HOST_DEVICE_API vec4(const vec3& _v)
        {
            v = _v;
            w = float(0);
        }

        inline AT_HOST_DEVICE_API operator vec3() const
        {
            return v;
        }
        inline AT_HOST_DEVICE_API float operator[](uint32_t i) const
        {
            AT_ASSERT(i < 4);
            return p[i];
        }
        inline AT_HOST_DEVICE_API float& operator[](uint32_t i)
        {
            AT_ASSERT(i < 4);
            return p[i];
        }

        AT_HOST_DEVICE_API vec4& operator=(const vec4& rhs)
        {
            v = rhs.v;
            w = rhs.w;
            return *this;
        }
        AT_HOST_DEVICE_API vec4& operator=(vec4&& rhs) noexcept
        {
            v = rhs.v;
            w = rhs.w;
            return *this;
        }

        inline AT_HOST_DEVICE_API const vec4& set(float _x, float _y, float _z, float _w)
        {
            x = _x;
            y = _y;
            z = _z;
            w = _w;
            return *this;
        }
        inline AT_HOST_DEVICE_API const vec4& set(float f)
        {
            x = f;
            y = f;
            z = f;
            w = f;
            return *this;
        }
        inline AT_HOST_DEVICE_API const vec4& set(const vec3& _v, float _w)
        {
            v = _v;
            w = _w;
            return *this;
        }

        inline AT_HOST_DEVICE_API const vec4& operator=(const vec3& rhs)
        {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            return *this;
        }

        inline AT_HOST_DEVICE_API const vec4& operator+() const
        {
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4 operator-() const
        {
            return vec4(-x, -y, -z, -w);
        }

        inline AT_HOST_DEVICE_API vec4& operator+=(const vec4& _v)
        {
            x += _v.x;
            y += _v.y;
            z += _v.z;
            w += _v.w;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4& operator-=(const vec4& _v)
        {
            x -= _v.x;
            y -= _v.y;
            z -= _v.z;
            w -= _v.w;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4& operator*=(const vec4& _v)
        {
            x *= _v.x;
            y *= _v.y;
            z *= _v.z;
            w *= _v.w;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4& operator/=(const vec4& _v)
        {
            x /= _v.x;
            y /= _v.y;
            z /= _v.z;
            w /= _v.w;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4& operator*=(const float t)
        {
            x *= t;
            y *= t;
            z *= t;
            w *= t;
            return *this;
        }
        inline AT_HOST_DEVICE_API vec4& operator/=(const float t)
        {
            x /= t;
            y /= t;
            z /= t;
            w /= t;
            return *this;
        }

        inline AT_HOST_DEVICE_API float length() const
        {
            auto ret = aten::sqrt(x * x + y * y + z * z);
            return ret;
        }
        inline AT_HOST_DEVICE_API float squared_length() const
        {
            auto ret = x * x + y * y + z * z;
            return ret;
        }

        inline AT_HOST_DEVICE_API void normalize()
        {
            auto invLen = aten::rsqrt(squared_length());
            *this *= invLen;
        }
    };

    inline AT_HOST_DEVICE_API vec4 operator+(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator+(const vec4 v1, float f)
    {
        vec4 ret(v1.x + f, v1.y + f, v1.z + f, v1.w + f);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator-(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator-(const vec4& v1, float f)
    {
        vec4 ret(v1.x - f, v1.y - f, v1.z - f, v1.w - f);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator*(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator/(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator*(float t, const vec4& v)
    {
        vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator*(const vec4& v, float t)
    {
        vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 operator/(const vec4& v, float t)
    {
        vec4 ret(v.x / t, v.y / t, v.z / t, v.w / t);
        return ret;
    }

    inline AT_HOST_DEVICE_API float dot(const vec4& v1, const vec4& v2)
    {
        auto ret = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 cross(const vec4& v1, const vec4& v2)
    {
        vec4 ret(
            v1.p[1] * v2.p[2] - v1.p[2] * v2.p[1],
            v1.p[2] * v2.p[0] - v1.p[0] * v2.p[2],
            v1.p[0] * v2.p[1] - v1.p[1] * v2.p[0],
            0);

        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 normalize(const vec4& v)
    {
        auto invLen = aten::rsqrt(dot(v, v));
        auto ret = v * invLen;
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 sqrt(const vec4& v)
    {
        vec4 ret(
            aten::sqrt(v.x),
            aten::sqrt(v.y),
            aten::sqrt(v.z),
            aten::sqrt(v.w));
        return ret;
    }

    inline AT_HOST_DEVICE_API vec4 abs(const vec4& v)
    {
        vec4 ret(
            aten::abs(v.x),
            aten::abs(v.y),
            aten::abs(v.z),
            aten::abs(v.w));
        return ret;
    }

    union _vec4_cmp_res {
        struct {
            uint8_t _0 : 1;
            uint8_t _1 : 1;
            uint8_t _2 : 1;
            uint8_t _3 : 1;
            uint8_t padding : 4;
        };
        uint8_t f;
    };

    // Compare Less EQual
    inline int32_t cmpLEQ(const vec4& a, const vec4& b)
    {
        _vec4_cmp_res res;

        res.f = 0;
        res._0 = (a.x <= b.x);
        res._1 = (a.y <= b.y);
        res._2 = (a.z <= b.z);
        res._3 = (a.w <= b.w);

        return res.f;
    }

    // Compare Greater EQual
    inline int32_t cmpGEQ(const vec4& a, const vec4& b)
    {
        _vec4_cmp_res res;

        res.f = 0;
        res._0 = (a.x >= b.x);
        res._1 = (a.y >= b.y);
        res._2 = (a.z >= b.z);
        res._3 = (a.w >= b.w);

        return res.f;
    }

    template <class T>
    inline AT_HOST_DEVICE_API T vmax(const T& a, const T& b)
    {
        return T(aten::max(a.x, b.x), aten::max(a.y, b.y), aten::max(a.z, b.z), aten::max(a.w, b.w));
    }

#ifdef __CUDACC__
    template <>
    inline AT_HOST_DEVICE_API float3 vmax(const float3& a, const float3& b)
    {
        return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    }

    template <>
    inline AT_HOST_DEVICE_API float4 vmax(const float4& a, const float4& b)
    {
        return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
    }
#endif

    template <class T>
    inline AT_HOST_DEVICE_API T vmin(const T& a, const T& b)
    {
        return T(aten::min(a.x, b.x), aten::min(a.y, b.y), aten::min(a.z, b.z), aten::min(a.w, b.w));
    }

#ifdef __CUDACC__
    template <>
    inline AT_HOST_DEVICE_API float3 vmin(const float3& a, const float3& b)
    {
        return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
    }

    template <>
    inline AT_HOST_DEVICE_API float4 vmin(const float4& a, const float4& b)
    {
        return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
    }
#endif
}
