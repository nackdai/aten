#pragma once

#include "defs.h"
#include "math/math.h"
#include "math/vec3.h"

namespace aten {
    class vec4 {
    public:
        union {
            vec3 v;
            struct {
                real x, y, z, w;
            };
            struct {
                real r, g, b, a;
            };
            real p[4];
        };

        AT_DEVICE_API vec4()
        {
            x = y = z = 0;
            w = 1;
        }
        AT_DEVICE_API vec4(const vec4& _v)
        {
            v = _v.v;
            w = _v.w;
        }
        AT_DEVICE_API vec4(real f)
        {
            x = y = z = w = f;
        }
        AT_DEVICE_API vec4(real _x, real _y, real _z, real _w)
        {
            x = _x;
            y = _y;
            z = _z;
            w = _w;
        }
        AT_DEVICE_API vec4(const vec3& _v, real _w)
        {
            v = _v;
            w = _w;
        }

        inline AT_DEVICE_API operator vec3() const
        {
            return v;
        }
        inline AT_DEVICE_API real operator[](uint32_t i) const
        {
            AT_ASSERT(i < 4);
            return p[i];
        }
        inline AT_DEVICE_API real& operator[](uint32_t i)
        {
            AT_ASSERT(i < 4);
            return p[i];
        }

        inline AT_DEVICE_API const vec4& set(real _x, real _y, real _z, real _w)
        {
            x = _x;
            y = _y;
            z = _z;
            w = _w;
            return *this;
        }
        inline AT_DEVICE_API const vec4& set(real f)
        {
            x = f;
            y = f;
            z = f;
            w = f;
            return *this;
        }
        inline AT_DEVICE_API const vec4& set(const vec3& _v, real _w)
        {
            v = _v;
            w = _w;
            return *this;
        }

        inline AT_DEVICE_API const vec4& operator=(const vec3& rhs)
        {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            return *this;
        }

        inline AT_DEVICE_API const vec4& operator+() const
        {
            return *this;
        }
        inline AT_DEVICE_API vec4 operator-() const
        {
            return vec4(-x, -y, -z, -w);
        }

        inline AT_DEVICE_API vec4& operator+=(const vec4& _v)
        {
            x += _v.x;
            y += _v.y;
            z += _v.z;
            w += _v.w;
            return *this;
        }
        inline AT_DEVICE_API vec4& operator-=(const vec4& _v)
        {
            x -= _v.x;
            y -= _v.y;
            z -= _v.z;
            w -= _v.w;
            return *this;
        }
        inline AT_DEVICE_API vec4& operator*=(const vec4& _v)
        {
            x *= _v.x;
            y *= _v.y;
            z *= _v.z;
            w *= _v.w;
            return *this;
        }
        inline AT_DEVICE_API vec4& operator/=(const vec4& _v)
        {
            x /= _v.x;
            y /= _v.y;
            z /= _v.z;
            w /= _v.w;
            return *this;
        }
        inline AT_DEVICE_API vec4& operator*=(const real t)
        {
            x *= t;
            y *= t;
            z *= t;
            w *= t;
            return *this;
        }
        inline AT_DEVICE_API vec4& operator/=(const real t)
        {
            x /= t;
            y /= t;
            z /= t;
            w /= t;
            return *this;
        }

        inline AT_DEVICE_API real length() const
        {
            auto ret = aten::sqrt(x * x + y * y + z * z);
            return ret;
        }
        inline AT_DEVICE_API real squared_length() const
        {
            auto ret = x * x + y * y + z * z;
            return ret;
        }

        inline AT_DEVICE_API void normalize()
        {
            auto invLen = aten::rsqrt(squared_length());
            *this *= invLen;
        }
    };

    inline AT_DEVICE_API vec4 operator+(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator+(const vec4 v1, real f)
    {
        vec4 ret(v1.x + f, v1.y + f, v1.z + f, v1.w + f);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator-(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator-(const vec4& v1, real f)
    {
        vec4 ret(v1.x - f, v1.y - f, v1.z - f, v1.w - f);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator*(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator/(const vec4& v1, const vec4& v2)
    {
        vec4 ret(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator*(real t, const vec4& v)
    {
        vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator*(const vec4& v, real t)
    {
        vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator/(const vec4& v, real t)
    {
        vec4 ret(v.x / t, v.y / t, v.z / t, v.w / t);
        return ret;
    }

    inline AT_DEVICE_API real dot(const vec4& v1, const vec4& v2)
    {
        auto ret = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
        return ret;
    }

    inline AT_DEVICE_API vec4 cross(const vec4& v1, const vec4& v2)
    {
        vec4 ret(
            v1.p[1] * v2.p[2] - v1.p[2] * v2.p[1],
            v1.p[2] * v2.p[0] - v1.p[0] * v2.p[2],
            v1.p[0] * v2.p[1] - v1.p[1] * v2.p[0],
            0);

        return ret;
    }

    inline AT_DEVICE_API vec4 normalize(const vec4& v)
    {
        auto invLen = aten::rsqrt(dot(v, v));
        auto ret = v * invLen;
        return ret;
    }

    inline AT_DEVICE_API vec4 sqrt(const vec4& v)
    {
        vec4 ret(
            aten::sqrt(v.x),
            aten::sqrt(v.y),
            aten::sqrt(v.z),
            aten::sqrt(v.w));
        return ret;
    }

    inline AT_DEVICE_API vec4 abs(const vec4& v)
    {
        vec4 ret(
            aten::abs(v.x),
            aten::abs(v.y),
            aten::abs(v.z),
            aten::abs(v.w));
        return ret;
    }

    inline AT_DEVICE_API vec4 min(const vec4& a, const vec4& b)
    {
        vec4 ret(
            aten::cmpMin(a.x, b.x),
            aten::cmpMin(a.y, b.y),
            aten::cmpMin(a.z, b.z),
            aten::cmpMin(a.w, b.w));
        return ret;
    }

    inline AT_DEVICE_API vec4 max(const vec4& a, const vec4& b)
    {
        vec4 ret(
            aten::cmpMax(a.x, b.x),
            aten::cmpMax(a.y, b.y),
            aten::cmpMax(a.z, b.z),
            aten::cmpMax(a.w, b.w));
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
}
