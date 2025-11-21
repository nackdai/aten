#pragma once

#include "types.h"
#include "math/vec3.h"

namespace aten {
    struct hitrecord {
        vec3 p;
        float area{ 0.0F };

        vec3 normal;
        int32_t mtrlid{ -1 };   ///< Index to the material.

        // texture coordinate.
        float u{ 0.0F };
        float v{ 0.0F };

        /**
         * @brief Index to the polygon group belonged to the same material.
         * This is unique in the entire scene.
         */
        int32_t meshid{ -1 };

        bool isVoxel{ false };
        uint8_t padding[3];
    };

    struct Intersection {
        float t{ AT_MATH_INF }; ///< Distance from the ray origin.

        int32_t objid{ -1 };    ///< Index to the root object.

        int32_t mtrlid{ -1 };   ///< Index to the material.

        /**
         * @brief Index to the polygon group belonged to the same material.
         * This is unique in the entire scene.
         */
        int32_t meshid{ -1 };

        union {
            // For triangle.
            struct {
                int32_t triangle_id;    ///< Hit triangle id.
                float a, b;             ///< Barycentric on hit triangle.
            };
            // Fox voxel.
            struct {
                float nml_x;
                float nml_y;
                float nml_z;
            };
        };

        int32_t isVoxel{ 0 };

        AT_HOST_DEVICE_API Intersection()
        {
            triangle_id = -1;
            a = b = float(0);

            nml_x = nml_y = nml_z = float(0);
        }
    };

    /**
     * @brief Result for sampled triangle.
     */
    struct SamplePosNormalPdfResult {
        aten::vec3 pos; ///< Sampled position.
        aten::vec3 nml; ///< Normal at sampled position.
        float area{ 0.0F }; ///< Area of sampled triangle.

        float a{ 0.0F };    ///< Barycentric on sampled triangle.
        float b{ 0.0F };    ///< Barycentric on sampled triangle.
        int32_t triangle_id{ -1 };  ///< Sampled triangle id.
    };
}
