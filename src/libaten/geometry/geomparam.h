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
    struct alignas(16) ObjectParameter {
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

        struct alignas(16) {
            vec3 center{ float(0) };    ///< Center of sphere.
            float radius{ 0 };          ///< Radius of sphere.
            int32_t mtrl_id{ -1 };      ///< Index of material.
        } sphere;
    };

    struct alignas(16) TriParam0 {
        int32_t idx[3]; ///< Vertex index.
        float padding;
    };

    struct alignas(16) TriParam1 {
        float area;     ///< Triangle area.
        int32_t needNormal{ 0 };    ///< Flag to describe if normal needs to be computed on the fly.
        int32_t mtrlid;             ///< Material id.
        /**
         * @brief Belonged mesh id. (Mesh = Triangle group to have same material).
         * This is unique in the entire scene.
         */
        int32_t mesh_id;
    };

    /**
     * @brief Parameter for per triangle.
     */
    struct alignas(16) TriangleParameter {
        TriParam0 v0;
        TriParam1 v1;

        static constexpr auto TriangleParamter_float4_size = (sizeof(TriParam0) + sizeof(TriParam1)) / 16;
        static AT_DEVICE_API TriParam0 ExtractTriParam0(const TriangleParameter* tris, size_t idx)
        {
            return (reinterpret_cast<aten::TriParam0*>(const_cast<aten::TriangleParameter*>(tris))[idx * TriangleParamter_float4_size + 0]);
        }
        static AT_DEVICE_API TriParam1 ExtractTriParam1(const TriangleParameter* tris, size_t idx)
        {
            return (reinterpret_cast<aten::TriParam1*>(const_cast<aten::TriangleParameter*>(tris))[idx * TriangleParamter_float4_size + 1]);
        }
    };
}
