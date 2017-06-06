#pragma once

#include "defs.h"
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
				// xŽ²‚Ì–Ê‚ÍŽè‘O‚Æ‰œ‚ª‚ ‚é‚Ì‚ÅA‚»‚ê‚¼‚ê‚Ì t ‚ðŒvŽZ.
				// t ‚ªxŽ²‚Ì–Ê‚ÌŽè‘O‚Æ‰œ‚Ì x ‚Ì”ÍˆÍ“à‚Å‚ ‚ê‚ÎAƒŒƒC‚ªAABB‚ð’Ê‚é.
				// ‚±‚ê‚ðxyzŽ²‚É‚Â‚¢‚ÄŒvŽZ‚·‚é.

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

		static aabb transform(const aabb& box, const aten::mat4& mtxL2W)
		{
			vec3 center = box.getCenter();

			vec3 vMin = box.minPos() - center;
			vec3 vMax = box.maxPos() - center;

			vec3 pts[8] = {
				make_float3(vMin.x, vMin.y, vMin.z),
				make_float3(vMax.x, vMin.y, vMin.z),
				make_float3(vMin.x, vMax.y, vMin.z),
				make_float3(vMax.x, vMax.y, vMin.z),
				make_float3(vMin.x, vMin.y, vMax.z),
				make_float3(vMax.x, vMin.y, vMax.z),
				make_float3(vMin.x, vMax.y, vMax.z),
				make_float3(vMax.x, vMax.y, vMax.z),
			};

			vec3 newMin = make_float3(AT_MATH_INF);
			vec3 newMax = make_float3(-AT_MATH_INF);

			for (int i = 0; i < 8; i++) {
				vec3 v = mtxL2W.apply(pts[i]);

				newMin = make_float3(
					std::min(newMin.x, v.x),
					std::min(newMin.y, v.y),
					std::min(newMin.z, v.z));
				newMax = make_float3(
					std::max(newMax.x, v.x),
					std::max(newMax.y, v.y),
					std::max(newMax.z, v.z));
			}

			aabb ret(newMin + center, newMax + center);

			return std::move(ret);
		}

	private:
		vec3 m_min;
		vec3 m_max;
	};
}
