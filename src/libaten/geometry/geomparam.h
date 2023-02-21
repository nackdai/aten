#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"
#include "material/material.h"

namespace aten
{
    enum class GeometryType : int32_t {
        Polygon,
        Instance,
        Sphere,
        Cube,
        GeometryTypeMax,
    };

    struct GeomParameter {
        GeometryType type{ GeometryType::GeometryTypeMax };

        real area{ real(0) };

        int32_t padding[2]{ 0, 0 };

        int32_t shapeid{ -1 };  ///< Own index in array.
        int32_t mtxid{ -1 };    ///< Index of matrix which the shape refer.
        int32_t primid{ -1 };   ///< First index of triangles which the shape has.
        int32_t primnum{ 0 };   ///< Number of triangles which the shape has.

        struct {
            int32_t idx{ -1 };
        } mtrl;

        vec3 center{ real(0) };

        vec3 size;          // cube.
        real radius{ 0 };   // shpere.

        GeomParameter() = default;
        ~GeomParameter() = default;
    };
    AT_STATICASSERT((sizeof(GeomParameter) % 16) == 0);

    struct PrimitiveParamter {
        union {
            aten::vec4 v0;
            struct {
                int32_t idx[3];
                real area;
            };
        };

        union {
            aten::vec4 v1;
            struct{
                int32_t needNormal;
                int32_t mtrlid;
                int32_t gemoid;
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
