#pragma once

#include "defs.h"
#include "math/math.h"

namespace aten {
    struct vec2 {
        float x;
        float y;

        AT_HOST_DEVICE_API vec2()
        {
            x = y = 0;
        }
        AT_HOST_DEVICE_API vec2(const vec2& _v)
        {
            x = _v.x;
            y = _v.y;
        }
        AT_HOST_DEVICE_API vec2(float f)
        {
            x = y = f;
        }
        AT_HOST_DEVICE_API vec2(float _x, float _y)
        {
            x = _x;
            y = _y;
        }
    };

    inline AT_HOST_DEVICE_API vec2 operator+(const vec2& v1, const vec2& v2)
    {
        vec2 ret(v1.x + v2.x, v1.y + v2.y);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec2 operator-(const vec2& v1, const vec2& v2)
    {
        vec2 ret(v1.x - v2.x, v1.y - v2.y);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec2 operator*(const vec2& v, float f)
    {
        vec2 ret(v.x * f, v.y * f);
        return ret;
    }

    inline AT_HOST_DEVICE_API vec2 operator/(const vec2& v1, const vec2& v2)
    {
        vec2 ret(v1.x / v2.x, v1.y / v2.y);
        return ret;
    }
}
