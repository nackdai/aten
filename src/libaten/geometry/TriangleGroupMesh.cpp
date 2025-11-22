#include <iterator>

#include "geometry/TriangleGroupMesh.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"
#include "geometry/vertex.h"

namespace AT_NAME
{
    float TriangleGroupMesh::build(context &ctxt, const std::optional<aten::vec3>& scale)
    {
        aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
        aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

        float area = 0;

        int32_t mtrlid = GetMaterial()->id();
        int32_t geomid = get_mesh_id();

        for (const auto f : triangles_)
        {
            f->build(ctxt, mtrlid, geomid, scale);

            const auto &faceParam = f->GetParam();
            area += faceParam.area;

            const auto &faabb = f->getBoundingbox();

            boxmin = aten::vmin(faabb.minPos(), boxmin);
            boxmax = aten::vmax(faabb.maxPos(), boxmax);
        }

        m_aabb.init(boxmin, boxmax);

        // For rasterize rendering.
        if (ctxt.IsWindowInitialized())
        {
            std::vector<uint32_t> idx;
            idx.reserve(triangles_.size() * 3);

            for (const auto f : triangles_)
            {
                const auto &faceParam = f->GetParam();

                idx.push_back(faceParam.idx[0]);
                idx.push_back(faceParam.idx[1]);
                idx.push_back(faceParam.idx[2]);
            }

            index_buffer_.init((uint32_t)idx.size(), &idx[0]);
        }

        return area;
    }

    void TriangleGroupMesh::AddFace(const std::shared_ptr<triangle> &f)
    {
        const auto &faceParam = f->GetParam();

        int32_t idx0 = faceParam.idx[0];
        int32_t idx1 = faceParam.idx[1];
        int32_t idx2 = faceParam.idx[2];

        triangles_.push_back(f);

        base_triangle_idx_ = std::min(f->GetId(), base_triangle_idx_);
    }

    void TriangleGroupMesh::render(
        aten::hitable::FuncPreDraw func,
        const context &ctxt,
        const aten::mat4 &mtx_L2W,
        const aten::mat4 &mtx_prev_L2W,
        int32_t parentId)
    {
        if (func)
        {
            func(mtx_L2W, mtx_prev_L2W, parentId, base_triangle_idx_);
        }

        const auto &vb = ctxt.GetVertexBuffer();

        auto triNum = (uint32_t)triangles_.size();

        index_buffer_.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }

    void TriangleGroupMesh::draw(
        AT_NAME::FuncObjectMeshDraw func,
        const context &ctxt)
    {
        if (func)
        {
            int32_t albedoTexId = mtrl_ ? mtrl_->param().albedoMap : -1;
            const auto albedo = albedoTexId >= 0 ? ctxt.GetTexture(albedoTexId) : nullptr;

            auto color = mtrl_ ? mtrl_->param().baseColor : vec4(1);

            auto mtrlid = mtrl_ ? mtrl_->id() : -1;

            func(color, albedo.get(), mtrlid);
        }

        const auto &vb = ctxt.GetVertexBuffer();

        auto triNum = (uint32_t)triangles_.size();

        index_buffer_.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }
}
