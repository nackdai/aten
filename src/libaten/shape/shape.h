#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"
#include "material/material.h"

namespace aten
{
	enum ShapeType : int {
		Polygon,
		Instance,
		Sphere,
		Cube,
		ShapeTypeMax,
	};

	struct ShapeParameter {
		ShapeType type{ ShapeType::ShapeTypeMax };

		real area;

		int shapeid{ -1 };	///< Own index in array.
		int mtxid{ -1 };	///< Index of matrix which the shape refer.
		int primid{ -1 };	///< First index of triangles which the shape has.
		int primnum{ 0 };	///< Number of triangles which the shape has.

		struct {
			vec3 center;
			union {
				vec3 size;		// cube.
				real radius;	// shpere.
			};
		};

		real padding[2];

		aten::UnionIdxPtr mtrl;

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
	AT_STATICASSERT((sizeof(ShapeParameter) % 16) == 0);

	struct PrimitiveParamter {
		int idx[3];
		int mtrlid;
		int needNormal{ 0 };
		real area;
		real padding[2];
	};
	AT_STATICASSERT((sizeof(PrimitiveParamter) % 16) == 0);
}
