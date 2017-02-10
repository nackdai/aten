#pragma once

#include "types.h"
#include "math/vec3.h"
#include "renderer/ray.h"
#include "material/material.h"

namespace aten
{
	class primitive;

	struct hitrecord {
		real t{ AT_MATH_INF };

		vec3 p;
		vec3 normal;

		primitive* obj{ nullptr };

		material* mtrl{ nullptr };
	};

	class primitive {
	protected:
		primitive() {}
		virtual ~primitive() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;
	};

	class sphere : public primitive {
	public:
		sphere() {}
		sphere(const vec3& c, real r, material* m)
			: m_center(c), m_radius(r), m_mtrl(m)
		{};

		virtual ~sphere() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		const vec3& center() const
		{
			return m_center;
		}

		real radius() const
		{
			return m_radius;
		}

	private:
		vec3 m_center;
		real m_radius{ CONST_REAL(0.0) };
		material* m_mtrl{ nullptr };
	};
}
