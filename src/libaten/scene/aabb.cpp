#include "aabb.h"

namespace aten {
	void aabb::init(const vec3& _min, const vec3& _max)
	{
		AT_ASSERT(_min.x <= _max.x);
		AT_ASSERT(_min.y <= _max.y);
		AT_ASSERT(_min.z <= _max.z);

		m_min = _min;
		m_max = _max;
	}

	void aabb::initBySize(const vec3& _min, const vec3& _size)
	{
		m_min = _min;
		m_max = m_min + _size;
	}

	vec3 aabb::size() const
	{
		vec3 size = m_max - m_min;
		return std::move(size);
	}

	bool aabb::hit(
		const ray& r,
		real t_min, real t_max) const
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
			// x軸の面は手前と奥があるので、それぞれの t を計算.
			// t がx軸の面の手前と奥の x の範囲内であれば、レイがAABBを通る.
			// これをxyz軸について計算する.

			auto t0 = (m_min[i] - r.org[i]) * inv;
			auto t1 = (m_max[i] - r.org[i]) * inv;

			if (inv < real(0)) {
				std::swap(t0, t1);
			}

			t_min = (t0 > t_min ? t0 : t_min);
			t_max = (t1 < t_max ? t1 : t_max);

			if (t_max <= t_min) {
				return false;
			}

			isHit = true;
		}

		return isHit;
	}

	bool aabb::isIn(const vec3& p) const
	{
		bool isInX = (m_min.x <= p.x && p.x <= m_max.x);
		bool isInY = (m_min.y <= p.y && p.y <= m_max.y);
		bool isInZ = (m_min.z <= p.z && p.z <= m_max.z);

		return isInX && isInY && isInZ;
	}

	bool aabb::isIn(const aabb& bound) const
	{
		bool b0 = isIn(bound.m_min);
		bool b1 = isIn(bound.m_max);

		return b0 & b1;
	}

	aabb aabb::surrounding_box(const aabb& box0, const aabb& box1)
	{
		vec3 _min(
			min(box0.m_min.x, box1.m_min.x),
			min(box0.m_min.y, box1.m_min.y),
			min(box0.m_min.z, box1.m_min.z));

		vec3 _max(
			max(box0.m_max.x, box1.m_max.x),
			max(box0.m_max.y, box1.m_max.y),
			max(box0.m_max.z, box1.m_max.z));

		aabb _aabb(_min, _max);

		return std::move(_aabb);
	}
}
