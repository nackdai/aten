#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"
#include "material/material.h"

namespace aten
{
    enum GeometryType : int {
        Polygon,
        Instance,
        Sphere,
        Cube,
        GeometryTypeMax,
    };

    struct GeomParameter {
        GeometryType type{ GeometryType::GeometryTypeMax };

        real area{ real(0) };

        int shapeid{ -1 };    ///< Own index in array.
        int mtxid{ -1 };    ///< Index of matrix which the shape refer.
        int primid{ -1 };    ///< First index of triangles which the shape has.
        int primnum{ 0 };    ///< Number of triangles which the shape has.

        vec3 center;
        union {
            vec3 size;        // cube.
            real radius;    // shpere.
        };

        real padding[2];

        aten::UnionIdxPtr mtrl;

        AT_DEVICE_API GeomParameter()
        {
            center = aten::vec3(real(0));
            mtrl.ptr = nullptr;
        }
        ~GeomParameter() {}
    };
    AT_STATICASSERT((sizeof(GeomParameter) % 16) == 0);

    struct PrimitiveParamter {
        union {
            aten::vec4 v0;
            struct {
                int idx[3];
                real area;
            };
        };
        
        union {
            aten::vec4 v1;
            struct{
                int needNormal;
                int mtrlid;
                int gemoid;
                real padding;
            };
        };

        AT_DEVICE_API PrimitiveParamter()
        {
            needNormal = 0;
        }

        AT_DEVICE_API PrimitiveParamter(const PrimitiveParamter& rhs)
        {
            v0 = rhs.v0;
            v1 = rhs.v1;
        }
    };
    AT_STATICASSERT((sizeof(PrimitiveParamter) % 16) == 0);

    const size_t PrimitiveParamter_float4_size = sizeof(PrimitiveParamter) / sizeof(aten::vec4);
}
