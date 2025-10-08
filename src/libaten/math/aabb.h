#pragma once

#include <limits>

#include "defs.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/mat4.h"
#include "math/ray.h"

namespace aten {
    class aabb {
    public:
        AT_HOST_DEVICE_API aabb()
        {
            empty();
        }
        AT_HOST_DEVICE_API aabb(const vec3& _min, const vec3& _max)
        {
            init(_min, _max);
        }
        AT_HOST_DEVICE_API ~aabb() {}

    public:
        AT_HOST_DEVICE_API void init(const vec3& _min, const vec3& _max)
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

        AT_HOST_DEVICE_API vec3 size() const
        {
            vec3 size = m_max - m_min;
            return size;
        }

        AT_HOST_DEVICE_API bool hit(
            const ray& r,
            float t_min, float t_max,
            float* t_result = nullptr) const
        {
            return hit(
                r,
                m_min, m_max,
                t_min, t_max,
                t_result);
        }

        template <class T>
        static AT_HOST_DEVICE_API bool hit(
            const ray& r,
            const T& _min, const T& _max,
            float t_min, float t_max,
            float* t_result = nullptr)
        {
            aten::vec3 invdir(float(1) / (r.dir + aten::vec3(float(1e-6))));

            auto oxinvdir = -r.org * invdir;

            const auto f = _max * invdir + oxinvdir;
            const auto n = _min * invdir + oxinvdir;

            const auto tmax = aten::vmax(f, n);
            const auto tmin = aten::vmin(f, n);

            const auto t1 = aten::min(aten::min_from_vec3(tmax), t_max);
            const auto t0 = aten::max(aten::max_from_vec3(tmin), t_min);

            if (t_result) {
                *t_result = t0;
            }

            return t0 <= t1;
        }

        template <class T>
        static AT_HOST_DEVICE_API bool hit(
            const ray& r,
            const T& _min, const T& _max,
            float t_min, float t_max,
            float& t_result,
            aten::vec3& nml)
        {
            bool is_hit = hit(r, _min, _max, t_min, t_max, &t_result);

            // NOTE
            // https://www.gamedev.net/forums/topic/551816-finding-the-aabb-surface-normal-from-an-intersection-point-on-aabb/

            auto point = r.org + t_result * r.dir;
            auto center = float(0.5) * (_min + _max);
            auto extent = float(0.5) * (_max - _min);

            // NOTE:
            // I can't overload glm::vec3::operator -=. So, extract -= to each element subtraction.
            point.x -= center.x;
            point.y -= center.y;
            point.z -= center.z;

            aten::vec3 sign(
                point.x < float(0) ? float(-1) : float(1),
                point.y < float(0) ? float(-1) : float(1),
                point.z < float(0) ? float(-1) : float(1));

            float minDist = AT_MATH_INF;

            auto dist = aten::abs(extent.x - aten::abs(point.x));
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

            return is_hit;
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

        AT_HOST_DEVICE_API const vec3& minPos() const
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

        AT_HOST_DEVICE_API vec3 GetCenter() const
        {
            vec3 center = (m_min + m_max) * 0.5F;
            return center;
        }

        static vec3 computeFaceSurfaceArea(
            const vec3& v_min,
            const vec3& v_max)
        {
            auto dx = aten::abs(v_max.x - v_min.x);
            auto dy = aten::abs(v_max.y - v_min.y);
            auto dz = aten::abs(v_max.z - v_min.z);

            return vec3(dy * dz, dz * dx, dx * dy);
        }

        vec3 computeFaceSurfaceArea() const
        {
            return computeFaceSurfaceArea(m_max, m_min);
        }

        static float computeSurfaceArea(
            const vec3& v_min,
            const vec3& v_max)
        {
            auto dx = aten::abs(v_max.x - v_min.x);
            auto dy = aten::abs(v_max.y - v_min.y);
            auto dz = aten::abs(v_max.z - v_min.z);

            // ６面の面積を計算するが、AABBは対称なので、３面の面積を計算して２倍すればいい.
            auto area = dx * dy;
            area += dy * dz;
            area += dz * dx;
            area *= 2;

            return area;
        }

        float computeSurfaceArea() const
        {
            return computeSurfaceArea(m_min, m_max);
        }

        AT_HOST_DEVICE_API void empty()
        {
            m_min.x = m_min.y = m_min.z = AT_MATH_INF;
            m_max.x = m_max.y = m_max.z = -AT_MATH_INF;
        }

        bool isEmpty() const
        {
            return m_min.x == AT_MATH_INF;
        }

        AT_HOST_DEVICE_API bool IsValid() const
        {
            return (aten::cmpGEQ(m_min, m_max) & 0x07) == 0;
        }

        AT_HOST_DEVICE_API float getDiagonalLenght() const
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
            vec3 _min = aten::vmin(m_min, v);
            vec3 _max = aten::vmax(m_max, v);

            m_min = _min;
            m_max = _max;
        }

        float volume() const
        {
            auto dx = aten::abs(m_max.x - m_min.x);
            auto dy = aten::abs(m_max.y - m_min.y);
            auto dz = aten::abs(m_max.z - m_min.z);

            return dx * dy * dz;
        }

        float computeRatio(const aabb& box)
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

        static aabb transform(const aabb& box, const aten::mat4& mtx_L2W)
        {
            vec3 center = box.GetCenter();

            vec3 v_min = box.minPos() - center;
            vec3 v_max = box.maxPos() - center;

            std::array pts = {
                vec3(v_min.x, v_min.y, v_min.z),
                vec3(v_max.x, v_min.y, v_min.z),
                vec3(v_min.x, v_max.y, v_min.z),
                vec3(v_max.x, v_max.y, v_min.z),
                vec3(v_min.x, v_min.y, v_max.z),
                vec3(v_max.x, v_min.y, v_max.z),
                vec3(v_min.x, v_max.y, v_max.z),
                vec3(v_max.x, v_max.y, v_max.z),
            };

            vec3 new_min = vec3(AT_MATH_INF);
            vec3 new_max = vec3(-AT_MATH_INF);

            for (int32_t i = 0; i < 8; i++) {
                vec3 v = mtx_L2W.apply(pts[i]);

                new_min = vec3(
                    std::min(new_min.x, v.x),
                    std::min(new_min.y, v.y),
                    std::min(new_min.z, v.z));
                new_max = vec3(
                    std::max(new_max.x, v.x),
                    std::max(new_max.y, v.y),
                    std::max(new_max.z, v.z));
            }

            aabb ret(new_min + center, new_max + center);

            return ret;
        }

        AT_HOST_DEVICE_API float ComputeSphereRadiusToCover() const
        {
            const auto center = GetCenter();
            const auto radius = length(m_max - center);
            return radius;
        }

        AT_HOST_DEVICE_API float ComputeDistanceToCoverBoundingSphere(float theta) const
        {
            // https://stackoverflow.com/questions/2866350/move-camera-to-fit-3d-scene
            const auto radius = ComputeSphereRadiusToCover();

            // r / d = tan(t/2) <=> d = r / tan(t/2)
            auto distance = radius / tan(theta / 2);

            return distance;
        }

    private:
        vec3 m_min{ std::numeric_limits<float>::max() };
        vec3 m_max{ std::numeric_limits<float>::min() };
    };
}
