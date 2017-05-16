#pragma once

#include "defs.h"
#include "math/vec3.h"
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
			AT_ASSERT(_min.x <= _max.x);
			AT_ASSERT(_min.y <= _max.y);
			AT_ASSERT(_min.z <= _max.z);

			m_min = _min;
			m_max = _max;
		}

		void initBySize(const vec3& _min, const vec3& _size)
		{
			m_min = _min;
			m_max = m_min + _size;
		}

		vec3 size() const
		{
			vec3 size = m_max - m_min;
			return std::move(size);
		}

		AT_DEVICE_API bool hit(
			const ray& r,
			real t_min, real t_max,
			real* t_result = nullptr) const
		{
			bool isHit = false;

			for (uint32_t i = 0; i < 3; i++) {
				if (r.dir[i] == 0.0f) {
					continue;
				}

				auto inv = real(1) / r.dir[i];

				// NOTE
				// ray : r = p + t * v
				// plane of AABB : x(t) = p(x) + t * v(x)
				//  t = (p(x) - x(t)) / v(x)
				// xŽ²‚Ì–Ê‚ÍŽè‘O‚Æ‰œ‚ª‚ ‚é‚Ì‚ÅA‚»‚ê‚¼‚ê‚Ì t ‚ðŒvŽZ.
				// t ‚ªxŽ²‚Ì–Ê‚ÌŽè‘O‚Æ‰œ‚Ì x ‚Ì”ÍˆÍ“à‚Å‚ ‚ê‚ÎAƒŒƒC‚ªAABB‚ð’Ê‚é.
				// ‚±‚ê‚ðxyzŽ²‚É‚Â‚¢‚ÄŒvŽZ‚·‚é.

				auto t0 = (m_min[i] - r.org[i]) * inv;
				auto t1 = (m_max[i] - r.org[i]) * inv;

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

		const vec3& minPos() const
		{
			return m_min;
		}

		const vec3& maxPos() const
		{
			return m_max;
		}

		vec3 getCenter() const
		{
			vec3 center = (m_min + m_max) * 0.5;
			return std::move(center);
		}

		real computeSurfaceArea() const
		{
			auto dx = aten::abs(m_max.x - m_min.x);
			auto dy = aten::abs(m_max.y - m_min.y);
			auto dz = aten::abs(m_max.z - m_min.z);

			// ‚U–Ê‚Ì–ÊÏ‚ðŒvŽZ‚·‚é‚ªAAABB‚Í‘ÎÌ‚È‚Ì‚ÅA‚R–Ê‚Ì–ÊÏ‚ðŒvŽZ‚µ‚Ä‚Q”{‚·‚ê‚Î‚¢‚¢.
			auto area = dx * dy;
			area += dy * dz;
			area += dz * dx;
			area *= 2;

			return area;
		}

		AT_DEVICE_API void empty()
		{
			m_min.x = m_min.y = m_min.z = AT_MATH_INF;
			m_max.x = m_max.y = m_max.z = -AT_MATH_INF;
		}

		static aabb merge(const aabb& box0, const aabb& box1)
		{
			vec3 _min = aten::make_float3(
				std::min(box0.m_min.x, box1.m_min.x),
				std::min(box0.m_min.y, box1.m_min.y),
				std::min(box0.m_min.z, box1.m_min.z));

			vec3 _max = aten::make_float3(
				std::max(box0.m_max.x, box1.m_max.x),
				std::max(box0.m_max.y, box1.m_max.y),
				std::max(box0.m_max.z, box1.m_max.z));

			aabb _aabb(_min, _max);

			return std::move(_aabb);
		}

	private:
		vec3 m_min;
		vec3 m_max;
	};
}
