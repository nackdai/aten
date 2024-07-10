#include "volume/grid.h"

#include "scene/host_scene_context.h"

namespace aten {
    void ConvertGridToMeshes(
        aten::context& ctxt,
        const Grid* grid)
    {
        const auto bbox = grid->worldBBox();
        const auto& bbox_min = bbox.min();
        const auto& bbox_max = bbox.max();

        constexpr int32_t IndicesPerTriangle = 3;
        constexpr int32_t TrianglesPerCubeFace = 2;
        constexpr int32_t FacesInCube = 6;
        constexpr int32_t IndicesInCube = IndicesPerTriangle * TrianglesPerCubeFace * FacesInCube;
        constexpr int32_t VerticesInCube = 8;

        /*
             4------------6
            /|           /|
           / |          / |
          5------------7  |
          |  0---------|--1
          | /          | /
          |/           |/
          2------------3
            +y
             |
             |
             +---> +x
            /
           +z
        */

        // Create triangles from bounding box
        std::vector<aten::vertex> vertices;
        vertices.reserve(VerticesInCube);
        {
            // Compute normal on the fly.
            // To do it, specify -1 at uv.z;
            aten::vertex v0{ aten::vec4(bbox_min[0], bbox_min[1], bbox_min[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v1{ aten::vec4(bbox_max[0], bbox_min[1], bbox_min[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v2{ aten::vec4(bbox_min[0], bbox_min[1], bbox_max[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v3{ aten::vec4(bbox_max[0], bbox_min[1], bbox_max[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };

            aten::vertex v4{ aten::vec4(bbox_min[0], bbox_max[1], bbox_min[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v5{ aten::vec4(bbox_min[0], bbox_max[1], bbox_max[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v6{ aten::vec4(bbox_max[0], bbox_max[1], bbox_min[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v7{ aten::vec4(bbox_max[0], bbox_max[1], bbox_max[2], 1), aten::vec3(), aten::vec3(0, 0, -1) };

            vertices.emplace_back(v0);
            vertices.emplace_back(v1);
            vertices.emplace_back(v2);
            vertices.emplace_back(v3);
            vertices.emplace_back(v4);
            vertices.emplace_back(v5);
            vertices.emplace_back(v6);
            vertices.emplace_back(v7);
        }

        std::vector<int32_t> indices;
        indices.reserve(IndicesInCube);
        {
            // bottom.
            indices.emplace_back(0);
            indices.emplace_back(3);
            indices.emplace_back(1);

            indices.emplace_back(0);
            indices.emplace_back(1);
            indices.emplace_back(2);

            // front.
            indices.emplace_back(2);
            indices.emplace_back(3);
            indices.emplace_back(7);

            indices.emplace_back(2);
            indices.emplace_back(7);
            indices.emplace_back(5);

            // right.
            indices.emplace_back(0);
            indices.emplace_back(2);
            indices.emplace_back(5);

            indices.emplace_back(0);
            indices.emplace_back(5);
            indices.emplace_back(4);

            // left.
            indices.emplace_back(3);
            indices.emplace_back(1);
            indices.emplace_back(6);

            indices.emplace_back(3);
            indices.emplace_back(6);
            indices.emplace_back(7);

            // back.
            indices.emplace_back(1);
            indices.emplace_back(0);
            indices.emplace_back(4);

            indices.emplace_back(1);
            indices.emplace_back(4);
            indices.emplace_back(6);

            // top.
            indices.emplace_back(5);
            indices.emplace_back(7);
            indices.emplace_back(6);

            indices.emplace_back(5);
            indices.emplace_back(6);
            indices.emplace_back(4);
        }


    }
}
