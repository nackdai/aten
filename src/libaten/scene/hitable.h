#pragma once 

#include <vector>
#include "types.h"
#include "scene/aabb.h"
#include "math/vec3.h"
#include "material/material.h"

namespace aten {
	class hitable;

	struct hitrecord {
		real t{ AT_MATH_INF };

		vec3 p;
		vec3 normal;
		vec3 u;
		vec3 v;

		hitable* obj{ nullptr };

		material* mtrl{ nullptr };
	};

	class hitable {
	public:
		hitable(const char* name = nullptr) : m_name(name) {}
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

		virtual aabb getBoundingbox() const = 0;

	private:
		const char* m_name;
	};
}
