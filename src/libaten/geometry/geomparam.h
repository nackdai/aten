#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"
#include "material/material.h"

namespace aten
{
    enum class ObjectType : int32_t {
        Polygon,    ///< Polygon.
        Instance,   ///< Instance. Not used in GPU.
        Sphere,     ///< Sphere.
        GeometryTypeMax,
    };

    struct ObjectParameter {
        ObjectType type{ ObjectType::GeometryTypeMax };

        /*
        * @brief Area of object
        *
        * - Polygon: Area of all triangles in this object.
        * - Instance: Not used.
        * - Sphere: Area of sphere.
        */
        real area{ real(0) };

        /**
        * @brief Own index.
        *
        * Meaning of this variable is changed based on geometry type.
        * - Polygon: Own index.
        * - Instance: Index of actual object which instance refers.
        * - Sphere: Not used.
        **/
        int32_t object_id{ -1 };

        int32_t mtx_id{ -1 };       ///< Index of matrix which geometry refers.

        int32_t triangle_id{ -1 };  ///< First index of triangles in geometry.
        uint32_t triangle_num{ 0 };  ///< Number of triangles in geometry.

        int32_t padding{ 0 };

        struct {
            vec3 center{ real(0) }; ///< Center of sphere.
            real radius{ 0 };       ///< Radius of sphere.
            int32_t mtrl_id{ -1 };  ///< Index of material.
        } sphere;

        ObjectParameter() = default;
        ~ObjectParameter() = default;
    };
    AT_STATICASSERT((sizeof(ObjectParameter) % 16) == 0);

    struct TriangleParameter {
        union {
            aten::vec4 v0;
            struct {
                int32_t idx[3];     ///< Vertex index.
                real area;          ///< Triangle area.
            };
        };

        union {
            aten::vec4 v1;
            struct{
                int32_t needNormal; ///< Flag to decribe if normal needs to be computed online.
                int32_t mtrlid;     ///< Material id.
                int32_t mesh_id;    ///< Mesh id. (Mesh = Triangle group to have same material).
                real padding;
            };
        };

        AT_DEVICE_API TriangleParameter()
        {
            needNormal = 0;
        }

        AT_DEVICE_API TriangleParameter(const TriangleParameter& rhs)
        {
            v0 = rhs.v0;
            v1 = rhs.v1;
        }
    };
    AT_STATICASSERT((sizeof(TriangleParameter) % 16) == 0);

    constexpr size_t TriangleParamter_float4_size = sizeof(TriangleParameter) / sizeof(aten::vec4);
}
