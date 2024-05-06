#pragma once

#include "vec4.h"
#include "mat4.h"

namespace aten
{
    template <class FType>
    class quaternion {
    public:
        union {
            FType a[4];
            struct {
                FType x, y, z, w;
            };
        };

        quaternion()
        {
            x = y = z = w = FType(0);
        }

        quaternion(float _x, float _y, float _z, float _w = 1.0f)
        {
            x = _x; y = _y; z = _z; w = _w;
        }

        quaternion(const quaternion& rhs)
        {
            x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w;
        }

        quaternion(const vec4& rhs)
        {
            x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w;
        }

        // 全成分０のクオータニオンを設定する
        quaternion& zero()
        {
            x = y = z = w = FType(0);
            return *this;
        }

        quaternion& identity()
        {
            x = y = z = FType(0);
            w = FType(1);
            return *this;
        }

        const quaternion& operator+() const
        {
            return *this;
        }
        quaternion operator-() const
        {
            return vec4(-x, -y, -z, -w);
        }

        quaternion& operator+=(const quaternion& _v)
        {
            x += _v.x;
            y += _v.y;
            z += _v.z;
            w += _v.w;
            return *this;
        }
        quaternion& operator-=(const quaternion& _v)
        {
            x -= _v.x;
            y -= _v.y;
            z -= _v.z;
            w -= _v.w;
            return *this;
        }
        quaternion& operator*=(const quaternion& _v)
        {
            *this = *this * _v;
            return *this;
        }
        quaternion& operator*=(float f)
        {
            *this = *this * f;
            return *this;
        }
        quaternion& operator/=(float f)
        {
            *this = *this / f;
            return *this;
        }

        // クオータニオンの大きさ(絶対値)を計算する
        float length()
        {
            float q = x * x + y * y + z * z + w * w;
            float ret = aten::sqrt(q);
            return ret;
        }

        // クオータニオンを正規化する
        quaternion& normalize()
        {
            auto l = length();
            *this /= l;
            return *this;
        }

        // 共役クオータニオンを求める
        quaternion& conjugate()
        {
            x = -x;
            y = -y;
            z = -z;
            w = w;
            return *this;
        }

        // 逆クオータニオンを求める
        quaternion& invert()
        {
            // |q|^2
            float s = x * x + y * y + z * z + w * w;

            // conjugate(q)
            conjugate();

            // q^-1 = conjugate(q) / |q|^2
            *this /= s;

            return *this;
        }

        // 球面線形補間
        static quaternion slerp(
            const quaternion& quat1,
            const quaternion& quat2,
            float t)
        {
            // NOTE
            // http://www.f-sp.com/entry/2017/06/30/221124#%E7%90%83%E9%9D%A2%E7%B7%9A%E5%BD%A2%E8%A3%9C%E9%96%93

            quaternion q1 = quat1;
            q1.normalize();

            quaternion q2 = quat2;
            q2.normalize();

            auto c = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
            c = aten::clamp(c, float(-1), float(1));

            auto theta = aten::acos(c);

            auto sdiv = aten::sin(theta);

            if (sdiv == float(0)) {
                return quat1;
            }

            auto s0 = aten::sin((float(1) - t) * theta);
            auto s1 = aten::sin(t * theta);

            quaternion ret = s0 / sdiv * quat1 + s1 / sdiv * quat2;

            return ret;
        }

        // 角度と任意軸からクオータニオンを設定する
        quaternion& setQuatFromRadAxis(float rad, const aten::vec4& vAxis)
        {
            // 念のため
            auto v = aten::normalize(vAxis);

            auto c = aten::cos(rad * float(0.5));
            auto s = aten::sin(rad * float(0.5));

            x = s * v.x;
            y = s * v.y;
            z = s * v.z;
            w = c;

            return *this;
        }

        // クオータニオンから行列を計算する
        mat4 getMatrix() const
        {
            mat4 mtx;

#if 1
            // ベタに計算するとこうなる
            const float xx2 = 2.0f * x * x;
            const float yy2 = 2.0f * y * y;
            const float zz2 = 2.0f * z * z;
            const float xy2 = 2.0f * x * y;
            const float xz2 = 2.0f * x * z;
            const float wz2 = 2.0f * w * z;
            const float wy2 = 2.0f * w * y;
            const float yz2 = 2.0f * y * z;
            const float wx2 = 2.0f * w * x;

            mtx.m[0][0] = 1.0f - yy2 - zz2;
            mtx.m[0][1] = xy2 + wz2;
            mtx.m[0][2] = xz2 - wy2;
            mtx.m[0][3] = 0.0f;

            mtx.m[1][0] = xy2 - wz2;
            mtx.m[1][1] = 1.0f - xx2 - zz2;
            mtx.m[1][2] = yz2 + wx2;
            mtx.m[1][3] = 0.0f;

            mtx.m[2][0] = xz2 + wy2;
            mtx.m[2][1] = yz2 - wx2;
            mtx.m[2][2] = 1.0f - xx2 - yy2;
            mtx.m[2][3] = 0.0f;

            mtx.m[3][0] = 0.0f;
            mtx.m[3][1] = 0.0f;
            mtx.m[3][2] = 0.0f;
            mtx.m[3][3] = 1.0f;

            // TODO
            mtx.transpose();
#else
            // マトリクスの乗算に直すとこうなる
            mat4 m1, m2;

            m1.m[0][0] = w; m1.m[0][1] = x; m1.m[0][2] = -y; m1.m[0][3] = z;
            m1.m[1][0] = -z; m1.m[1][1] = y; m1.m[1][2] = x; m1.m[1][3] = w;
            m1.m[2][0] = y; m1.m[2][1] = z; m1.m[2][2] = w; m1.m[2][3] = -x;
            m1.m[3][0] = m1.m[3][1] = m1.m[3][2] = m1.m[3][3] = 0.0f;

            m2.m[0][0] = w; m2.m[0][1] = z; m2.m[0][2] = -y; m2.m[0][3] = 0.0f;
            m2.m[1][0] = x; m2.m[1][1] = y; m2.m[1][2] = z; m2.m[1][3] = 0.0f;
            m2.m[2][0] = y; m2.m[2][1] = -x; m2.m[2][2] = w; m2.m[2][3] = 0.0f;
            m2.m[3][0] = -z; m2.m[3][1] = w; m2.m[3][2] = x; m2.m[3][3] = 0.0f;

            mtx = m1 * m2;
            mtx.m[3][3] = 1.0f;
#endif

            return mtx;
        }

        // 行列からクオータニオンを計算する
        quaternion& fromMatrix(const mat4& mtx)
        {
            // 最大値を探す
            float value[4] = {
                mtx.m[0][0] - mtx.m[1][1] - mtx.m[2][2] + float(1),
                -mtx.m[0][0] + mtx.m[1][1] - mtx.m[2][2] + float(1),
                -mtx.m[0][0] - mtx.m[1][1] + mtx.m[2][2] + float(1),
                mtx.m[0][0] + mtx.m[1][1] + mtx.m[2][2] + float(1),
            };

            uint32_t nMaxValIdx = 0;
            for (uint32_t i = 0; i < 4; ++i) {
                if (value[i] > value[nMaxValIdx]) {
                    nMaxValIdx = i;
                }
            }

            AT_ASSERT(value[nMaxValIdx] > float(0));

            float v = sqrtf(value[nMaxValIdx]) * float(0.5);

            // NOTE
            // 1 / (4 * v)
            float div = float(0.25) / v;

            switch (nMaxValIdx) {
            case 0:    // x
                x = v;
                y = (mtx.m[0][1] + mtx.m[1][0]) * div;
                z = (mtx.m[2][0] + mtx.m[0][2]) * div;
                w = (mtx.m[1][2] - mtx.m[2][1]) * div;
                break;
            case 1:    // y
                x = (mtx.m[0][1] + mtx.m[1][0]) * div;
                y = v;
                z = (mtx.m[1][2] + mtx.m[2][1]) * div;
                w = (mtx.m[2][0] - mtx.m[0][2]) * div;
                break;
            case 2:    // z
                x = (mtx.m[2][0] + mtx.m[0][2]) * div;
                y = (mtx.m[1][2] + mtx.m[2][1]) * div;
                z = v;
                w = (mtx.m[0][1] - mtx.m[1][0]) * div;
                break;
            case 3:    // w
                x = (mtx.m[1][2] - mtx.m[2][1]) * div;
                y = (mtx.m[2][0] - mtx.m[0][2]) * div;
                z = (mtx.m[0][1] - mtx.m[1][0]) * div;
                w = v;
                break;
            }

            return *this;
        }

        quaternion& fromEuler(const aten::vec3& euler)
        {
            return fromEuler(euler.x, euler.y, euler.z);
        }

        // オイラー角からクオータニオンを計算する
        quaternion& fromEuler(float x, float y, float z)
        {
            // Q1 = (x1, y1, z1, w1) = x1i + y1j + z1k + w1
            // Q2 = (x2, y2, z2, w2) = x2i + y2j + z2k + w2
            //
            // ii = jj = kk = -1
            // ij = -ji = k
            // jk = -kj = i
            // ki = -ik = j
            //
            // Q1Q2 = (w1w2 - x1x2 - y1y2 - z1-z2)
            //        + (w1x2 + x1w2 + y1z2 - z1y2)i
            //        + (w1y2 - x1z2 + y1w2 + z1x2)j
            //        + (w1z2 + x1y2 - y1x2 + z1w2)k

            // Qyaw   = (0,            sin(yaw/2), 0,           cos(yaw/2))
            // Qpitch = (sin(pitch/2), 0,          0,           cos(pitch/2))
            // Qroll  = (0,            0,          sin(roll/2), cos(roll/2))
            //
            // Q = QyawQpitchQroll -> Roll Pitch Yaw の順番
            //  (qPq^-1 = QyawQpitchQrollPq^-1 からRoll Pitch Yaw の順番でかけることになる）
            //
            // Cy = cos(yaw/2), Cp = cos(pitch/2), Cr = cos(roll/2)
            // Sy = sin(yaw/2), Sp = sin(pitch/2), Sr = sin(roll/2)
            //
            // QpitchQroll = (CpCr) + (SpCr)i + (-SpSR)j + (CpSr)k
            // QyawQpirchQroll = (CyCpCr + SySpSr)
            //                      + (CySpCr + SyCpCr)i
            //                      + (-CySpSr + SyCpCr)j
            //                      + (CyCpSr - SySpCr)k

            // Yaw
            float cosY = aten::cos(y * float(0.5));
            float sinY = aten::sin(y * float(0.5));

            // Pitch
            float cosX = aten::cos(x * float(0.5));
            float sinX = aten::sin(x * float(0.5));

            // Roll
            float cosZ = aten::cos(z * float(0.5));
            float sinZ = aten::sin(z * float(0.5));

            x = cosZ * sinX * cosY + sinZ * cosX * sinY;
            y = cosZ * cosX * sinY - sinZ * sinX * cosY;
            z = sinZ * cosX * cosY - cosZ * sinX * sinY;
            w = cosZ * cosX * cosY + sinZ * sinX * sinY;

            return *this;
        }

        // 二つのベクトルv0,v1が与えられたときに
        // q  * v0 == v1 となるクオータニオンqを計算する
        static quaternion RotationArc(const vec4& from, const vec4& to)
        {
            // 念のため
            vec4 v0 = aten::normalize(from);
            vec4 v1 = aten::normalize(to);

            // http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm

            // angle = arcos(v1•v2 / | v1 || v2 | )
            // axis = norm(v1 x v2)
            // s = sin(angle / 2)
            // x = axis.x *s
            // y = axis.y *s
            // z = axis.z *s
            // w = cos(angle / 2)

            vec4 axis = cross(v0, v1);

            float cosine = dot(v0, v1);
            float angle = acosf(cosine);

            float s = aten::sin(angle * float(0.5));

            quaternion ret;
            ret.x = axis.x * s;
            ret.y = axis.y * s;
            ret.z = axis.z * s;
            ret.w = aten::cos(angle * float(0.5));

            return ret;
        }

        // クオータニオンからオイラー角を計算する
        vec3 getEuler()
        {
            auto mtx = getMatrix();

            // TODO
            // MatrixFromQuatの結果はZXYの回転順番であることを常に保障する？

            // NOTE
            // m[0][0] = CzCy + SzSxSy  m[0][1] = SzCx m[0][2] = -CzSy + SzSxCy m[0][3] = 0
            // m[1][0] = -SzCy + CzSxSy m[1][1] = CzCx m[1][2] = SzSy + CzSxCy  m[1][3] = 0
            // m[2][0] = CxSy           m[2][1] = -Sx  m[2][2] = CxCy           m[2][3] = 0
            // m[3][0] = 0              m[3][1] = 0    m[3][2] = 0              m[3][3] = 1

            float Sx = -mtx.m[2][1];
            float Cx = aten::sqrt(float(1) - Sx * Sx);

            vec3 angle;

            if (Cx != 0.0f)
            {
                angle.x = ::acosf(Cx);

                float Sy = mtx.m[2][0] / Cx;
                float Cy = mtx.m[2][2] / Cx;
                angle.y = ::atan2f(Sy, Cy);

                float Sz = mtx.m[0][1] / Cx;
                float Cz = mtx.m[1][1] / Cx;
                angle.z = ::atan2f(Sz, Cz);
            }
            else
            {
                if (Cx > 0.0f)
                {
                    // Xの回転角度が -90度(=270度)
                    angle.x = Deg2Rad(-90.0f);
                }
                else
                {
                    // Xの回転角度が 90度
                    angle.x = Deg2Rad(90.0f);
                }

                angle.y = 0.0f;

                float Cz = mtx.m[0][0];
                float Sz = -mtx.m[1][0];
                angle.z = ::atan2f(Sz, Cz);
            }

            return angle;
        }
    };

    template <class FType>
    inline quaternion<FType> operator+(const quaternion<FType>& v1, const quaternion<FType>& v2)
    {
        quaternion<FType> ret(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
        return ret;
    }

    template <class FType>
    inline quaternion<FType> operator-(const quaternion<FType>& v1, const quaternion<FType>& v2)
    {
        quaternion<FType> ret(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
        return ret;
    }

    template <class FType>
    inline quaternion<FType> operator*(const quaternion<FType>& v1, const quaternion<FType>& v2)
    {
        quaternion<FType> dst;

        dst.x = v1.w * v2.x + v2.w * v1.x + v1.y * v2.z - v1.z * v2.y;
        dst.y = v1.w * v2.y + v2.w * v1.y + v1.z * v2.x - v1.x * v2.z;
        dst.z = v1.w * v2.z + v2.w * v1.z + v1.x * v2.y - v1.y * v2.x;

        dst.w = v1.w * v2.w - v1.x * v2.x - v1.y * v2.y - v1.z * v2.z;

        return dst;
    }

    template <class FType>
    inline quaternion<FType> operator*(const quaternion<FType>& v, float t)
    {
        quaternion<FType> ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    template <class FType>
    inline quaternion<FType> operator*(float t, const quaternion<FType>& v)
    {
        quaternion<FType> ret(t * v.x, t * v.y, t * v.z, t * v.w);
        return ret;
    }

    template <class FType>
    inline quaternion<FType> operator/(const quaternion<FType>& v, float t)
    {
        quaternion<FType> ret(v.x / t, v.y / t, v.z / t, v.w / t);
        return ret;
    }

    using quat = quaternion<float>;
}
