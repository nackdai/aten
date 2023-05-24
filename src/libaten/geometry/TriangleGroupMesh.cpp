#include <iterator>

#include "geometry/TriangleGroupMesh.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"
#include "geometry/vertex.h"
#include "visualizer/window.h"

namespace AT_NAME
{
    TriangleGroupMesh::~TriangleGroupMesh()
    {
        triangles_.clear();
    }

    float TriangleGroupMesh::build(const context& ctxt)
    {
        aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
        aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

        float area = 0;

        int32_t mtrlid = getMaterial()->id();
        int32_t geomid = get_mesh_id();

        for (const auto f : triangles_) {
            f->build(ctxt, mtrlid, geomid);

            const auto& faceParam = f->getParam();
            area += faceParam.area;

            const auto& faabb = f->getBoundingbox();

            boxmin = aten::vmin(faabb.minPos(), boxmin);
            boxmax = aten::vmax(faabb.maxPos(), boxmax);
        }

        m_aabb.init(boxmin, boxmax);

        // For rasterize rendering.
        if (window::isInitialized())
        {
            std::vector<uint32_t> idx;
            idx.reserve(triangles_.size() * 3);

            for (const auto f : triangles_) {
                const auto& faceParam = f->getParam();

                idx.push_back(faceParam.idx[0]);
                idx.push_back(faceParam.idx[1]);
                idx.push_back(faceParam.idx[2]);
            }

            m_ib.init((uint32_t)idx.size(), &idx[0]);
        }

        return area;
    }

    void TriangleGroupMesh::addFace(const std::shared_ptr<triangle>& f)
    {
        const auto& faceParam = f->getParam();

        int32_t idx0 = faceParam.idx[0];
        int32_t idx1 = faceParam.idx[1];
        int32_t idx2 = faceParam.idx[2];

        triangles_.push_back(f);

        m_baseTriIdx = std::min(f->getId(), m_baseTriIdx);
    }

    void TriangleGroupMesh::render(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int32_t parentId)
    {
        if (func) {
            func(mtxL2W, mtxPrevL2W, parentId, m_baseTriIdx);
        }

        const auto& vb = ctxt.getVB();

        auto triNum = (uint32_t)triangles_.size();

        m_ib.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }

    void TriangleGroupMesh::draw(
        AT_NAME::FuncObjectMeshDraw func,
        const context& ctxt)
    {
        if (func) {
            int32_t albedoTexId = m_mtrl ? m_mtrl->param().albedoMap : -1;
            const auto albedo = albedoTexId >= 0 ? ctxt.getTexture(albedoTexId) : nullptr;

            auto color = m_mtrl ? m_mtrl->param().baseColor : vec3(1);

            auto mtrlid = m_mtrl ? m_mtrl->id() : -1;

            func(color, albedo.get(), mtrlid);
        }

        const auto& vb = ctxt.getVB();

        auto triNum = (uint32_t)triangles_.size();

        m_ib.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }
}
