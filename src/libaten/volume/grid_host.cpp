#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#pragma warning(pop)

#include "geometry/transformable_factory.h"
#include "scene/host_scene_context.h"
#include "volume/grid.h"

namespace aten {
    std::shared_ptr<aten::PolygonObject> GenerateTrianglesFromGridBoundingBox(
        aten::context& ctxt,
        const int32_t target_mtrl_id,
        const nanovdb::FloatGrid* grid)
    {
        constexpr int32_t IndicesPerTriangle = 3;
        constexpr int32_t TrianglesPerCubeFace = 2;
        constexpr int32_t FacesInCube = 6;
        constexpr int32_t IndicesInCube = IndicesPerTriangle * TrianglesPerCubeFace * FacesInCube;
        constexpr int32_t VerticesInCube = 8;

        const auto grid_bbox = grid->worldBBox();
        const auto& bbox_min = grid_bbox.min();
        const auto& bbox_max = grid_bbox.max();

        const auto triangle_index_offset = ctxt.GetVertexNum();

        /*
         *     4------------6
         *    /|           /|
         *   / |          / |
         *  5------------7  |
         *  |  0---------|--1
         *  | /          | /
         *  |/           |/
         *  2------------3
         *    +y
         *     |
         *     |
         *     +---> +x
         *    /
         *   +z
         */

        // Create triangles from bounding box
        {
            // Compute normal on the fly.
            // To do it, specify -1 at uv.z;
            aten::vertex v0{ aten::vec4(bbox_min[0], bbox_min[1], bbox_min[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v1{ aten::vec4(bbox_max[0], bbox_min[1], bbox_min[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v2{ aten::vec4(bbox_max[0], bbox_max[1], bbox_min[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v3{ aten::vec4(bbox_min[0], bbox_max[1], bbox_min[2]), aten::vec3(), aten::vec3(0, 0, -1) };

            aten::vertex v4{ aten::vec4(bbox_max[0], bbox_min[1], bbox_max[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v5{ aten::vec4(bbox_min[0], bbox_min[1], bbox_max[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v6{ aten::vec4(bbox_min[0], bbox_max[1], bbox_max[2]), aten::vec3(), aten::vec3(0, 0, -1) };
            aten::vertex v7{ aten::vec4(bbox_max[0], bbox_max[1], bbox_max[2]), aten::vec3(), aten::vec3(0, 0, -1) };

            ctxt.AddVertex(v0);
            ctxt.AddVertex(v1);
            ctxt.AddVertex(v2);
            ctxt.AddVertex(v3);
            ctxt.AddVertex(v4);
            ctxt.AddVertex(v5);
            ctxt.AddVertex(v6);
            ctxt.AddVertex(v7);
        }

        constexpr std::array TriangleIndices = {
            // back.
            0, 2, 1,
            0, 3, 2,

            // right.
            1, 7, 4,
            1, 2, 7,

            // front.
            4, 7, 6,
            4, 6, 5,

            // left.
            5, 3, 0,
            5, 6, 3,

            // floor.
            0, 1, 4,
            0, 4, 5,

            // top.
            3, 7, 2,
            3, 6, 7,
        };

        auto mesh = std::make_shared<aten::TriangleGroupMesh>();

        const auto mtrl = ctxt.GetMaterialInstance(target_mtrl_id);
        mesh->SetMaterial(mtrl);

        for (size_t i = 0; i < TriangleIndices.size(); i += 3)
        {
            aten::TriangleParameter tri;
            tri.idx[0] = TriangleIndices[i] + triangle_index_offset;
            tri.idx[1] = TriangleIndices[i + 1] + triangle_index_offset;
            tri.idx[2] = TriangleIndices[i + 2] + triangle_index_offset;
            tri.mtrlid = target_mtrl_id;
            tri.mesh_id = mesh->get_mesh_id();
            tri.needNormal = true;

            auto face = ctxt.CreateTriangle(tri);

            mesh->AddFace(face);
        }

        auto obj = aten::TransformableFactory::createObject(ctxt);
        obj->appendShape(mesh);

        aten::aabb bbox(
            aten::vec3(bbox_min[0], bbox_min[1], bbox_min[2]),
            aten::vec3(bbox_max[0], bbox_max[1], bbox_max[2]));
        obj->setBoundingBox(bbox);

        return obj;
    }
}
