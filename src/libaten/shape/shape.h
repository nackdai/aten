#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"

namespace aten
{
	struct vertex {
		vec3 pos;
		vec3 nml;
		vec3 uv;
	};

	struct ShapeParameter {
		union {
			// triangle.
			struct {
				aabb bbox;
				int idx[3];
				vertex* vtx[3];
				real area;
			};
			// sphere.
			struct {
				aabb bbox;
				vec3 center;
				real radius;
			};
			// cube.
			struct {
				aabb bbox;
				vec3 center;
				vec3 size;
			};
		};

		ShapeParameter() {}

		// sphere.
		ShapeParameter(const vec3& c, real r)
			: center(c), radius(r)
		{
			vec3 _min = center - radius;
			vec3 _max = center + radius;

			bbox.init(_min, _max);
		}

		// cube.
		ShapeParameter(const vec3& c, const vec3& s)
			: center(c), size(s)
		{
			bbox.init(
				center - size * 0.5,
				center + size * 0.5);
		}

		~ShapeParameter() {}
	};
}
