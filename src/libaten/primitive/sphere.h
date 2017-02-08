#pragma once

#include "types.h"
#include "math/vec3.h"
#include "renderer/ray.h"
#include "material/material.h"

namespace aten
{
	struct hitrecord {
		real t{ CONST_REAL(0.0) };

		vec3 p;
		vec3 normal;

		material* mtrl{ nullptr };
	};

	class sphere {
	public:
		sphere() {}
		sphere(const vec3& c, real r, material* m)
			: m_center(c), m_radius(r), m_mtrl(m)
		{};

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const;

	private:
		vec3 m_center;
		real m_radius{ CONST_REAL(0.0) };
		material* m_mtrl{ nullptr };
	};
}
