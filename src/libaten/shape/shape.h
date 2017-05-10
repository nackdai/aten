#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"
#include "material/material.h"

namespace aten
{
	enum ShapeType {
		Polygon,
		Instance,
		Sphere,
		Cube,
		ShapeTypeMax,
	};

	struct ShapeParameter {
		ShapeType type{ ShapeType::ShapeTypeMax };

		aten::UnionIdxPtr mtrl;

		mat4 mtxW2L;
		mat4 mtxL2W;

		real area;

		int primid{ -1 };
		int shapeid{ -1 };

		struct {
			vec3 center;
			union {
				vec3 size;		// cube.
				real radius;	// shpere.
			};
		};

		AT_DEVICE_API ShapeParameter(ShapeType _type)
			: type(_type)
		{}

		// sphere.
		AT_DEVICE_API ShapeParameter(const vec3& c, real r, AT_NAME::material* m)
			: center(c), radius(r), type(ShapeType::Sphere)
		{
			mtrl.ptr = m;
		}

		// cube.
		AT_DEVICE_API ShapeParameter(const vec3& c, const vec3& s, AT_NAME::material* m)
			: center(c), size(s), type(ShapeType::Sphere)
		{
			mtrl.ptr = m;
		}

		~ShapeParameter() {}
	};

	struct PrimitiveParamter {
		int idx[3];
		int mtrlid;
		real area;
	};
}
