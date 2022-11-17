#pragma once

#include "defs.h"
#include "math/math.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/ray.h"

namespace aten {
    class mat4 {
    public:
        static const mat4 Identity;
        static const mat4 Zero;

        union {
            real a[16];
            real m[4][4];
            vec4 v[4];
            struct {
                real m00, m01, m02, m03;
                real m10, m11, m12, m13;
                real m20, m21, m22, m23;
                real m30, m31, m32, m33;
            };
        };

        AT_DEVICE_API mat4()
        {
            identity();
        }
        AT_DEVICE_API mat4(const mat4& rhs)
        {
            *this = rhs;
        }
        AT_DEVICE_API mat4(
            real _m00, real _m01, real _m02, real _m03,
            real _m10, real _m11, real _m12, real _m13,
            real _m20, real _m21, real _m22, real _m23,
            real _m30, real _m31, real _m32, real _m33)
        {
            m00 = _m00; m01 = _m01; m02 = _m02; m03 = _m03;
            m10 = _m10; m11 = _m11; m12 = _m12; m13 = _m13;
            m20 = _m20; m21 = _m21; m22 = _m22; m23 = _m23;
            m30 = _m30; m31 = _m31; m32 = _m32; m33 = _m33;
        }
        AT_DEVICE_API mat4(const vec3& x, const vec3& y, const vec3& z)
        {
            m00 = x.x; m01 = y.x; m02 = z.x; m03 = 0;
            m10 = x.y; m11 = y.y; m12 = z.y; m13 = 0;
            m20 = x.z; m21 = y.z; m22 = z.z; m23 = 0;
            m30 = 0.0; m31 = 0.0; m32 = 0.0; m33 = 1;
        }

        inline AT_DEVICE_API mat4& identity()
        {
            // TODO
#ifdef __AT_CUDA__
            m00 = 1; m01 = 0; m02 = 0; m03 = 0;
            m10 = 0; m11 = 1; m12 = 0; m13 = 0;
            m20 = 0; m21 = 0; m22 = 1; m23 = 0;
            m30 = 0; m31 = 0; m32 = 0; m33 = 1;
#else
            *this = Identity;
#endif
            return *this;
        }

        inline bool isIdentity() const
        {
            return (m00 == 1 && m01 == 0 && m02 == 0 && m03 == 0
            && m10 == 0 && m11 == 1 && m12 == 0 && m13 == 0
            && m20 == 0 && m21 == 0 && m22 == 1 && m23 == 0
            && m30 == 0 && m31 == 0 && m32 == 0 && m33 == 1);
        }

        inline mat4& zero()
        {
            *this = Zero;
            return *this;
        }

        inline const real* data() const
        {
            return a;
        }

        inline AT_DEVICE_API const mat4& operator+() const
        {
            return *this;
        }

        inline AT_DEVICE_API mat4 operator-() const
        {
            mat4 ret;
            ret.m00 = -m00; ret.m01 = -m01; ret.m02 = -m02; ret.m03 = -m03;
            ret.m10 = -m10; ret.m11 = -m11; ret.m12 = -m12; ret.m13 = -m13;
            ret.m20 = -m20; ret.m21 = -m21; ret.m22 = -m22; ret.m23 = -m23;
            ret.m30 = -m30; ret.m31 = -m31; ret.m32 = -m32; ret.m33 = -m33;
            return ret;
        }

        inline AT_DEVICE_API real* operator[](int i)
        {
            return m[i];
        }
        inline AT_DEVICE_API real operator()(int i, int j) const
        {
            return m[i][j];
        }
        inline AT_DEVICE_API real& operator()(int i, int j)
        {
            return m[i][j];
        }

        inline AT_DEVICE_API mat4& operator+=(const mat4& mtx)
        {
            m00 += mtx.m00; m01 += mtx.m01; m02 += mtx.m02; m03 += mtx.m03;
            m10 += mtx.m10; m11 += mtx.m11; m12 += mtx.m12; m13 += mtx.m13;
            m20 += mtx.m20; m21 += mtx.m21; m22 += mtx.m22; m23 += mtx.m23;
            m30 += mtx.m30; m31 += mtx.m31; m32 += mtx.m32; m33 += mtx.m33;
            return *this;
        }
        inline AT_DEVICE_API mat4& operator-=(const mat4& mtx)
        {
            m00 -= mtx.m00; m01 -= mtx.m01; m02 -= mtx.m02; m03 -= mtx.m03;
            m10 -= mtx.m10; m11 -= mtx.m11; m12 -= mtx.m12; m13 -= mtx.m13;
            m20 -= mtx.m20; m21 -= mtx.m21; m22 -= mtx.m22; m23 -= mtx.m23;
            m30 -= mtx.m30; m31 -= mtx.m31; m32 -= mtx.m32; m33 -= mtx.m33;
            return *this;
        }
        inline AT_DEVICE_API mat4& operator*=(const mat4& mtx)
        {
            mat4 tmp;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    tmp.m[i][j] = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        tmp.m[i][j] += this->m[i][k] * mtx.m[k][j];
                    }
                }
            }

            *this = tmp;

            return *this;
        }

        inline AT_DEVICE_API mat4& operator*=(const real t)
        {
            m00 *= t; m01 *= t; m02 *= t; m03 *= t;
            m10 *= t; m11 *= t; m12 *= t; m13 *= t;
            m20 *= t; m21 *= t; m22 *= t; m23 *= t;
            m30 *= t; m31 *= t; m32 *= t; m33 *= t;
            return *this;
        }
        inline AT_DEVICE_API mat4& operator/=(const real t)
        {
            *this *= 1 / t;
            return *this;
        }

        inline AT_DEVICE_API vec4 apply(const vec4& p) const
        {
            vec4 ret;
            ret.x = v[0].x * p.x + v[0].y * p.y + v[0].z * p.z + v[0].w * p.w;
            ret.y = v[1].x * p.x + v[1].y * p.y + v[1].z * p.z + v[1].w * p.w;
            ret.z = v[2].x * p.x + v[2].y * p.y + v[2].z * p.z + v[2].w * p.w;
            ret.w = v[3].x * p.x + v[3].y * p.y + v[3].z * p.z + v[3].w * p.w;

            return ret;
        }

        inline AT_DEVICE_API vec3 apply(const vec3& p) const
        {
#if 0
            vec4 t(v.x, v.y, v.z, 1);
            vec4 ret;

            for (int r = 0; r < 4; r++) {
                ret[r] = 0;
                for (int c = 0; c < 4; c++) {
                    ret[r] += m[r][c] * t[c];
                }
            }
#else
            vec3 ret;
            ret.x = v[0].x * p.x + v[0].y * p.y + v[0].z * p.z + v[0].w;
            ret.y = v[1].x * p.x + v[1].y * p.y + v[1].z * p.z + v[1].w;
            ret.z = v[2].x * p.x + v[2].y * p.y + v[2].z * p.z + v[2].w;
#endif

            return ret;
        }
        inline AT_DEVICE_API vec3 applyXYZ(const vec3& p) const
        {
            vec3 ret;

#if 0
            for (int r = 0; r < 3; r++) {
                ret[r] = 0;
                for (int c = 0; c < 3; c++) {
                    ret[r] += m[r][c] * v[c];
                }
            }
#else
            ret.x = v[0].x * p.x + v[0].y * p.y + v[0].z * p.z;
            ret.y = v[1].x * p.x + v[1].y * p.y + v[1].z * p.z;
            ret.z = v[2].x * p.x + v[2].y * p.y + v[2].z * p.z;
#endif

            return ret;
        }

        inline ray applyRay(const ray& r) const
        {
            vec3 org = r.org;
            vec3 dir = r.dir;

            // Transform world to local.
            org = apply(org);
            dir = applyXYZ(dir);

            ray transformdRay(org, dir);

            return transformdRay;
        }

        mat4& invert();

        inline AT_DEVICE_API mat4& transpose()
        {
            mat4 tmp;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    tmp.m[i][j] = this->m[j][i];
                }
            }

            *this = tmp;

            return *this;
        }

        inline AT_DEVICE_API mat4& asTrans(const vec3& v)
        {
            identity();

            m03 = v.x;
            m13 = v.y;
            m23 = v.z;

            return *this;
        }

        inline AT_DEVICE_API mat4& asTrans(real x, real y, real z)
        {
            return asTrans(vec3(x, y, z));
        }

        inline AT_DEVICE_API mat4& asScale(real s)
        {
            identity();

            m00 = s;
            m11 = s;
            m22 = s;

            return *this;
        }

        inline AT_DEVICE_API mat4& asRotateByX(real r)
        {
            const real c = aten::cos(r);
            const real s = aten::sin(r);

            m00 = 1; m01 = 0; m02 = 0;  m03 = 0;
            m10 = 0; m11 = c; m12 = -s; m13 = 0;
            m20 = 0; m21 = s; m22 = c;  m23 = 0;
            m30 = 0; m31 = 0; m32 = 0;  m33 = 1;

            return *this;
        }

        inline AT_DEVICE_API mat4& asRotateByY(real r)
        {
            const real c = aten::cos(r);
            const real s = aten::sin(r);

            m00 = c;  m01 = 0; m02 = s; m03 = 0;
            m10 = 0;  m11 = 1; m12 = 0; m13 = 0;
            m20 = -s; m21 = 0; m22 = c; m23 = 0;
            m30 = 0;  m31 = 0; m32 = 0; m33 = 1;

            return *this;
        }

        inline AT_DEVICE_API mat4& asRotateByZ(real r)
        {
            const real c = aten::cos(r);
            const real s = aten::sin(r);

            m00 = c; m01 = -s; m02 = 0; m03 = 0;
            m10 = s; m11 = c;  m12 = 0; m13 = 0;
            m20 = 0; m21 = 0;  m22 = 1; m23 = 0;
            m30 = 0; m31 = 0;  m32 = 0; m33 = 1;

            return *this;
        }

        mat4& asRotateByAxis(real r, const vec3& axis);

        mat4& asRotateFromVector(const vec3& v, const vec3& up);

        inline mat4& asScale(const vec3& v)
        {
            m00 = v.x; m01 = 0;   m02 = 0;   m03 = 0;
            m10 = 0;   m11 = v.y; m12 = 0;   m13 = 0;
            m20 = 0;   m21 = 0;   m22 = v.z; m23 = 0;
            m30 = 0;   m31 = 0;   m32 = 0;   m33 = 1;

            return *this;
        }

        inline mat4& asScale(real x, real y, real z)
        {
            return asScale(vec3(x, y, z));
        }

        mat4& lookat(
            const vec3& eye,
            const vec3& at,
            const vec3& up);

        mat4& perspective(
            real znear, real zfar,
            real vfov,
            real aspect);

        mat4& ortho(
            real width, real height,
            real znear, real zfar);

        void dump() const
        {
            AT_PRINTF("%f %f %f %f\n", m[0][0], m[0][1], m[0][2], m[0][3]);
            AT_PRINTF("%f %f %f %f\n", m[1][0], m[1][1], m[1][2], m[1][3]);
            AT_PRINTF("%f %f %f %f\n", m[2][0], m[2][1], m[2][2], m[2][3]);
            AT_PRINTF("%f %f %f %f\n", m[3][0], m[3][1], m[3][2], m[3][3]);
        }
    };

    inline AT_DEVICE_API mat4 operator+(const mat4& m1, const mat4& m2)
    {
        mat4 ret = m1;
        ret += m2;
        return ret;
    }

    inline AT_DEVICE_API mat4 operator-(const mat4& m1, const mat4& m2)
    {
        mat4 ret = m1;
        ret -= m2;
        return ret;
    }

    inline AT_DEVICE_API mat4 operator*(const mat4& m1, const mat4& m2)
    {
        mat4 ret = m1;
        ret *= m2;
        return ret;
    }

    inline AT_DEVICE_API vec3 operator*(const mat4& m, const vec3 v)
    {
        vec3 ret = m.apply(v);
        return ret;
    }

    inline AT_DEVICE_API vec4 operator*(const mat4& m, const vec4 v)
    {
        vec4 ret = m.apply(v);
        return ret;
    }

    inline AT_DEVICE_API mat4 operator*(real t, const mat4& m)
    {
        mat4 ret = m;
        ret *= t;
        return ret;
    }

    inline AT_DEVICE_API mat4 operator*(const mat4& m, real t)
    {
        mat4 ret = t * m;
        return ret;
    }

    inline AT_DEVICE_API mat4 operator/(const mat4& m, real t)
    {
        mat4 ret = m * (1 / t);
        return ret;
    }
}
