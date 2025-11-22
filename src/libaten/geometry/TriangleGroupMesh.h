#pragma once

#include <vector>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/triangle.h"
#include "visualizer/GeomDataBuffer.h"
#include "scene/host_scene_context.h"

namespace AT_NAME
{
    using FuncObjectMeshDraw = std::function<void(const aten::vec3&, const aten::texture*, int32_t)>;

    /**
     * @brief Triangle group to have the same material.
     */
    class TriangleGroupMesh : public aten::NoHitableMesh {
        friend class PolygonObject;

    public:
        TriangleGroupMesh() = default;
        virtual ~TriangleGroupMesh()
        {
            triangles_.clear();
        }

        TriangleGroupMesh(const TriangleGroupMesh&) = delete;
        TriangleGroupMesh(TriangleGroupMesh&&) = delete;
        TriangleGroupMesh& operator=(const TriangleGroupMesh&) = delete;
        TriangleGroupMesh& operator=(TriangleGroupMesh&&) = delete;

        /**
         * @brief Build TriangleGroupMesh.
         *
         * @param[in] ctxt Instance of scene context.
         * @param[in] scale Value to scale triangles and those vertices. If this is invalid, nothing happens.
         * @return Area of this mesh.
         */
        float build(aten::context& ctxt, const std::optional<aten::vec3>& scale);

        /**
         * @breif Set a material for the traiangle group.
         *
         * @param mtrl Material to be specified for the triangle group.
         */
        void SetMaterial(const std::shared_ptr<material>& mtrl)
        {
            mtrl_ = mtrl;
        }

        /**
         * @brief Get a material of the triangle group.
         *
         * @return Material of the triangle group as const reference.
         */
        const std::shared_ptr<material>& GetMaterial() const
        {
            return mtrl_;
        }

        /**
         * @brief Get a material of the triangle group.
         *
         * @return Material of the triangle group.
         */
        std::shared_ptr<material> GetMaterial()
        {
            return mtrl_;
        }

        /**
         * @brief Add a triangle.
         */
        void AddFace(const std::shared_ptr<triangle>& f);

        void render(
            aten::hitable::FuncPreDraw func,
            const aten::context& ctxt,
            const aten::mat4& mtx_L2W,
            const aten::mat4& mtx_prev_L2W,
            int32_t parentId);

        void draw(
            AT_NAME::FuncObjectMeshDraw func,
            const aten::context& ctxt);

        /**
         * @brief Get all triangles in the triangle group as list.
         *
         * @return Triangle list in the triangle group.
         */
        const std::vector<std::shared_ptr<triangle>>& GetTriangleList() const
        {
            return triangles_;
        }

        aten::aabb m_aabb;

    private:
        std::shared_ptr<material> mtrl_;
        std::vector<std::shared_ptr<triangle>> triangles_;

        aten::GeomIndexBuffer index_buffer_;

        int32_t base_triangle_idx_{ INT32_MAX };
    };
}
