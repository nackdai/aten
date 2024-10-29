#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/aabb.h"
#include "math/mat4.h"

namespace aten
{
    /**
     * @brief Object type.
     */
    enum class ObjectType : int32_t {
        Polygons,   ///< Polygons.
        Instance,   ///< Instance. Not used in GPU.
        Sphere,     ///< Sphere.
        GeometryTypeMax,
    };

    /**
     * @brief Parameter for object.
     */
    struct ObjectParameter {
        ObjectType type{ ObjectType::GeometryTypeMax };

        /**
         * @brief Area of object
         *
         * - Polygons: Area of all triangles in this object.
         * - Instance: Not used.
         * - Sphere: Area of sphere.
         */
        float area{ float(0) };

        /**
         * @brief Own index.
         *
         * Meaning of this variable is changed based on geometry type.
         * - Polygons: Not used.
         * - Instance: Index of actual object which instance refers.
         * - Sphere: Not used.
         */
        int32_t object_id{ -1 };

        /**
         * @brief Index of matrix to apply to object.
         *
         * The matrix is converting from world to local.
         * mtx_id + 1 indicates the inverse matrix to convert from local to world.
         * This is used for only Instance type.
         */
        int32_t mtx_id{ -1 };

        int32_t triangle_id{ -1 };  ///< First index of triangles in object.
        int32_t triangle_num{ 0 };  ///< Number of triangles in object.

        int32_t light_id{ -1 };     ///< If there is an associated light, index to light.

        struct {
            vec3 center{ float(0) };    ///< Center of sphere.
            float radius{ 0 };          ///< Radius of sphere.
            int32_t mtrl_id{ -1 };      ///< Index of material.
        } sphere;
    };
    AT_STATICASSERT((sizeof(ObjectParameter) % 16) == 0);

    /**
     * @brief Parameter for per triangle.
     */
    struct TriangleParameter {
        union {
            aten::vec4 v0;
            struct {
                int32_t idx[3]; ///< Vertex index.
                float area;     ///< Triangle area.
            };
        };

        union {
            aten::vec4 v1;
            struct {
                int32_t needNormal; ///< Flag to describe if normal needs to be computed on the fly.
                int32_t mtrlid;     ///< Material id.
                int32_t mesh_id;    ///< Belonged mesh id. (Mesh = Triangle group to have same material).
                float padding;
            };
        };

        AT_HOST_DEVICE_API TriangleParameter()
        {
            needNormal = 0;
        }

        AT_HOST_DEVICE_API TriangleParameter(const TriangleParameter& rhs)
        {
            v0 = rhs.v0;
            v1 = rhs.v1;
        }

        AT_HOST_DEVICE_API TriangleParameter& operator=(const TriangleParameter& rhs)
        {
            v0 = rhs.v0;
            v1 = rhs.v1;
            return *this;
        }
    };
    AT_STATICASSERT((sizeof(TriangleParameter) % 16) == 0);

    constexpr size_t TriangleParamter_float4_size = sizeof(TriangleParameter) / sizeof(aten::vec4);
}
