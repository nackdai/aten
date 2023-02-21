#pragma once

#include "defs.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/mat4.h"
#include "math/ray.h"

namespace aten {
    class aabb {
    public:
        AT_DEVICE_API aabb()
        {
            empty();
        }
        AT_DEVICE_API aabb(const vec3& _min, const vec3& _max)
        {
            init(_min, _max);
        }
        AT_DEVICE_API ~aabb() {}

    public:
        AT_DEVICE_API void init(const vec3& _min, const vec3& _max)
        {
#if 0
            AT_ASSERT(_min.x <= _max.x);
            AT_ASSERT(_min.y <= _max.y);
            AT_ASSERT(_min.z <= _max.z);
#endif

            m_min = _min;
            m_max = _max;
        }

        void initBySize(const vec3& _min, const vec3& _size)
        {
            m_min = _min;
            m_max = m_min + _size;
        }

        AT_DEVICE_API vec3 size() const
        {
            vec3 size = m_max - m_min;
            return size;
        }

        AT_DEVICE_API bool hit(
            const ray& r,
            real t_min, real t_max,
            real* t_result = nullptr) const
        {
            return hit(
                r,
                m_min, m_max,
                t_min, t_max,
                t_result);
        }

        static AT_DEVICE_API bool hit(
            const ray& r,
            const aten::vec3& _min, const aten::vec3& _max,
            real t_min, real t_max,
            real* t_result = nullptr)
        {
#if 0
            bool isHit = false;

            for (uint32_t i = 0; i < 3; i++) {
                if (_min[i] == _max[i]) {
                    continue;
                }

#if 0
                if (r.dir[i] == 0.0f) {
                    continue;
                }

                auto inv = real(1) / r.dir[i];
#else
                auto inv = real(1) / (r.dir[i] + real(1e-6));
#endif

                // NOTE
                // ray : r = p + t * v
                // plane of AABB : x(t) = p(x) + t * v(x)
                //  t = (p(x) - x(t)) / v(x)
                // x軸の面は手前と奥があるので、それぞれの t を計算.
                // t がx軸の面の手前と奥の x の範囲内であれば、レイがAABBを通る.
                // これをxyz軸について計算する.

                auto t0 = (_min[i] - r.org[i]) * inv;
                auto t1 = (_max[i] - r.org[i]) * inv;

                if (inv < real(0)) {
#if 0
                    std::swap(t0, t1);
#else
                    real tmp = t0;
                    t0 = t1;
                    t1 = tmp;
#endif
                }

                t_min = (t0 > t_min ? t0 : t_min);
                t_max = (t1 < t_max ? t1 : t_max);

                if (t_max <= t_min) {
                    return false;
                }

                if (t_result) {
                    *t_result = t0;
                }

                isHit = true;
            }

            return isHit;
#else
            aten::vec3 invdir = real(1) / (r.dir + aten::vec3(real(1e-6)));
            aten::vec3 oxinvdir = -r.org * invdir;

            const auto f = _max * invdir + oxinvdir;
            const auto n = _min * invdir + oxinvdir;

            const auto tmax = max(f, n);
            const auto tmin = min(f, n);

            const auto t1 = aten::cmpMin(aten::cmpMin(aten::cmpMin(tmax.x, tmax.y), tmax.z), t_max);
            const auto t0 = aten::cmpMax(aten::cmpMax(aten::cmpMax(tmin.x, tmin.y), tmin.z), t_min);

            if (t_result) {
                *t_result = t0;
            }

            return t0 <= t1;
#endif
        }

        static AT_DEVICE_API bool hit(
            const ray& r,
            const aten::vec3& _min, const aten::vec3& _max,
            real t_min, real t_max,
            real& t_result,
            aten::vec3& nml)
        {
            bool isHit = hit(r, _min, _max, t_min, t_max, &t_result);

            // NOTE
            // https://www.gamedev.net/forums/topic/551816-finding-the-aabb-surface-normal-from-an-intersection-point-on-aabb/

            auto point = r.org + t_result * r.dir;
            auto center = real(0.5) * (_min + _max);
            auto extent = real(0.5) * (_max - _min);

            point -= center;

            aten::vec3 sign(
                point.x < real(0) ? real(-1) : real(1),
                point.y < real(0) ? real(-1) : real(1),
                point.z < real(0) ? real(-1) : real(1));

            real minDist = AT_MATH_INF;

            real dist = aten::abs(extent.x - aten::abs(point.x));
            if (dist < minDist) {
                minDist = dist;
                nml = sign.x * aten::vec3(1, 0, 0);
            }

            dist = aten::abs(extent.y - aten::abs(point.y));
            if (dist < minDist) {
                minDist = dist;
                nml = sign.y * aten::vec3(0, 1, 0);
            }

            dist = aten::abs(extent.z - aten::abs(point.z));
            if (dist < minDist) {
                minDist = dist;
                nml = sign.z * aten::vec3(0, 0, 1);
            }

            return isHit;
        }

        bool isIn(const vec3& p) const
        {
            bool isInX = (m_min.x <= p.x && p.x <= m_max.x);
            bool isInY = (m_min.y <= p.y && p.y <= m_max.y);
            bool isInZ = (m_min.z <= p.z && p.z <= m_max.z);

            return isInX && isInY && isInZ;
        }

        bool isIn(const aabb& bound) const
        {
            bool b0 = isIn(bound.m_min);
            bool b1 = isIn(bound.m_max);

            return b0 & b1;
        }

        AT_DEVICE_API const vec3& minPos() const
        {
            return m_min;
        }

        vec3& minPos()
        {
            return m_min;
        }

        const vec3& maxPos() const
        {
            return m_max;
        }

        vec3& maxPos()
        {
            return m_max;
        }

        vec3 getCenter() const
        {
            vec3 center = (m_min + m_max) * real(0.5);
            return center;
        }

        static vec3 computeFaceSurfaceArea(
            const vec3& vMin,
            const vec3& vMax)
        {
            auto dx = aten::abs(vMax.x - vMin.x);
            auto dy = aten::abs(vMax.y - vMin.y);
            auto dz = aten::abs(vMax.z - vMin.z);

            return vec3(dy * dz, dz * dx, dx * dy);
        }

        vec3 computeFaceSurfaceArea() const
        {
            return computeFaceSurfaceArea(m_max, m_min);
        }

        static real computeSurfaceArea(
            const vec3& vMin,
            const vec3& vMax)
        {
            auto dx = aten::abs(vMax.x - vMin.x);
            auto dy = aten::abs(vMax.y - vMin.y);
            auto dz = aten::abs(vMax.z - vMin.z);

            // ６面の面積を計算するが、AABBは対称なので、３面の面積を計算して２倍すればいい.
            auto area = dx * dy;
            area += dy * dz;
            area += dz * dx;
            area *= 2;

            return area;
        }

        real computeSurfaceArea() const
        {
            return computeSurfaceArea(m_min, m_max);
        }

        AT_DEVICE_API void empty()
        {
            m_min.x = m_min.y = m_min.z = AT_MATH_INF;
            m_max.x = m_max.y = m_max.z = -AT_MATH_INF;
        }

        bool isEmpty() const
        {
            return m_min.x == AT_MATH_INF;
        }

        bool isValid() const
        {
            return (aten::cmpGEQ(m_min, m_max) & 0x07) == 0;
        }

        AT_DEVICE_API real getDiagonalLenght() const
        {
            auto ret = length(m_max - m_min);
            return ret;
        }

        void expand(const aabb& box)
        {
            *this = merge(*this, box);
        }

        void expand(const vec3& v)
        {
            vec3 _min = aten::min(m_min, v);
            vec3 _max = aten::max(m_max, v);

            m_min = _min;
            m_max = _max;
        }

        real volume() const
        {
            auto dx = aten::abs(m_max.x - m_min.x);
            auto dy = aten::abs(m_max.y - m_min.y);
            auto dz = aten::abs(m_max.z - m_min.z);

            return dx * dy * dz;
        }

        real computeRatio(const aabb& box)
        {
            auto v0 = volume();
            auto v1 = box.volume();

            return v1 / (v0 + AT_MATH_EPSILON);
        }

        void merge(const aabb& b)
        {
            if (isEmpty()) {
                m_min = b.m_min;
                m_max = b.m_max;
            }
            else {
                m_min.x = std::min(m_min.x, b.m_min.x);
                m_min.y = std::min(m_min.y, b.m_min.y);
                m_min.z = std::min(m_min.z, b.m_min.z);

                m_max.x = std::max(m_max.x, b.m_max.x);
                m_max.y = std::max(m_max.y, b.m_max.y);
                m_max.z = std::max(m_max.z, b.m_max.z);
            }
        }

        static aabb merge(const aabb& box0, const aabb& box1)
        {
            vec3 _min = aten::vec3(
                std::min(box0.m_min.x, box1.m_min.x),
                std::min(box0.m_min.y, box1.m_min.y),
                std::min(box0.m_min.z, box1.m_min.z));

            vec3 _max = aten::vec3(
                std::max(box0.m_max.x, box1.m_max.x),
                std::max(box0.m_max.y, box1.m_max.y),
                std::max(box0.m_max.z, box1.m_max.z));

            aabb _aabb(_min, _max);

            return _aabb;
        }

        static aabb transform(const aabb& box, const aten::mat4& mtxL2W)
        {
            vec3 center = box.getCenter();

            vec3 vMin = box.minPos() - center;
            vec3 vMax = box.maxPos() - center;

            vec3 pts[8] = {
                vec3(vMin.x, vMin.y, vMin.z),
                vec3(vMax.x, vMin.y, vMin.z),
                vec3(vMin.x, vMax.y, vMin.z),
                vec3(vMax.x, vMax.y, vMin.z),
                vec3(vMin.x, vMin.y, vMax.z),
                vec3(vMax.x, vMin.y, vMax.z),
                vec3(vMin.x, vMax.y, vMax.z),
                vec3(vMax.x, vMax.y, vMax.z),
            };

            vec3 newMin = vec3(AT_MATH_INF);
            vec3 newMax = vec3(-AT_MATH_INF);

            for (int32_t i = 0; i < 8; i++) {
                vec3 v = mtxL2W.apply(pts[i]);

                newMin = vec3(
                    std::min(newMin.x, v.x),
                    std::min(newMin.y, v.y),
                    std::min(newMin.z, v.z));
                newMax = vec3(
                    std::max(newMax.x, v.x),
                    std::max(newMax.y, v.y),
                    std::max(newMax.z, v.z));
            }

            aabb ret(newMin + center, newMax + center);

            return ret;
        }

    private:
        vec3 m_min;
        vec3 m_max;
    };
}
