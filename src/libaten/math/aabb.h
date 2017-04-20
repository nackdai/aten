#pragma once

#include "defs.h"
#include "math/vec3.h"
#include "math/ray.h"

namespace aten {
	class aabb {
	public:
		aabb()
		{
			empty();
		}
		aabb(const vec3& _min, const vec3& _max)
		{
			init(_min, _max);
		}
		~aabb() {}

	public:
		void init(const vec3& _min, const vec3& _max);

		void initBySize(const vec3& _min, const vec3& _size);

		vec3 size() const;

		bool hit(
			const ray& r,
			real t_min, real t_max,
			real* t_result = nullptr) const;

		bool isIn(const vec3& p) const;

		bool isIn(const aabb& bound) const;

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

		real computeSurfaceArea() const;

		void empty();

		static aabb merge(const aabb& box0, const aabb& box1);

	private:
		vec3 m_min;
		vec3 m_max;
	};
}
