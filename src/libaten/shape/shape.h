#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "material/material.h"

namespace aten
{
	struct vertex {
		vec3 pos;
		vec3 nml;
		vec3 uv;
	};

	enum ShapeType {
		Triangle,
		Sphere,
		Cube,
	};

	struct ShapeParameter {
		ShapeType type{ ShapeType::Triangle };
		aabb bbox;

		aten::UnionIdxPtr mtrl;

		union {
			// triangle.
			struct {
				int idx[3];
				real area;
			};
			// sphere / cube.
			struct {
				vec3 center;
				union {
					vec3 size;		// cube.
					real radius;	// shpere.
				};
			};
		};

		AT_DEVICE_API ShapeParameter() {}

		// sphere.
		AT_DEVICE_API ShapeParameter(const vec3& c, real r, AT_NAME::material* m)
			: center(c), radius(r), type(ShapeType::Sphere)
		{
			vec3 _min = center - radius;
			vec3 _max = center + radius;

			bbox.init(_min, _max);

			mtrl.ptr = m;
		}

		// cube.
		AT_DEVICE_API ShapeParameter(const vec3& c, const vec3& s, AT_NAME::material* m)
			: center(c), size(s), type(ShapeType::Sphere)
		{
			bbox.init(
				center - size * 0.5,
				center + size * 0.5);

			mtrl.ptr = m;
		}

		~ShapeParameter() {}
	};
}
