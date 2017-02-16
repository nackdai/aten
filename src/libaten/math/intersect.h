#pragma once

#include "defs.h"
#include "math/vec3.h"
#include "math/ray.h"

namespace aten {
	struct intersectResult {
		bool isIntersect{ false };
		real t;
	};

	inline intersectResult intersertTriangle(
		const ray& ray,
		const vec3& v0,
		const vec3& v1,
		const vec3& v2)
	{
		// NOTE
		// http://qiita.com/edo_m18/items/2bd885b13bd74803a368
		// http://kanamori.cs.tsukuba.ac.jp/jikken/inner/triangle_intersection.pdf

		vec3 e1 = v1 - v0;
		vec3 e2 = v2 - v0;
		vec3 r = ray.org - v0;
		vec3 d = ray.dir;

		vec3 u = cross(d, e2);
		vec3 v = cross(r, e1);

		real inv = CONST_REAL(1.0) / dot(u, e1);

		real t = dot(v, e2) * inv;
		real beta = dot(u, r) * inv;
		real gamma = dot(v, d) * inv;

		intersectResult result;

		result.isIntersect = ((beta >= CONST_REAL(0.0) && beta <= CONST_REAL(1.0))
			&& (gamma >= CONST_REAL(0.0) && gamma <= CONST_REAL(1.0))
			&& (beta + gamma <= CONST_REAL(1.0))
			&& t >= CONST_REAL(0.0));

		result.t = t;

		return std::move(result);
	}
}
