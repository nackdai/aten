#include <iterator>

#include "geometry/objshape.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"
#include "geometry/vertex.h"
#include "visualizer/window.h"

namespace AT_NAME
{
    objshape::~objshape()
    {
        faces.clear();
    }

    void objshape::build(const context& ctxt)
    {
        aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
        aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

        param.area = 0;

        int mtrlid = getMaterial()->id();
        int geomid = getGeomId();

        for (const auto f : faces) {
            f->build(ctxt, mtrlid, geomid);

            const auto& faceParam = f->getParam();
            param.area += faceParam.area;

            const auto& faabb = f->getBoundingbox();

            boxmin = aten::min(faabb.minPos(), boxmin);
            boxmax = aten::max(faabb.maxPos(), boxmax);
        }

        m_aabb.init(boxmin, boxmax);

        // For rasterize rendering.
        if (window::isInitialized())
        {
            std::vector<uint32_t> idx;
            idx.reserve(faces.size() * 3);

            for (const auto f : faces) {
                const auto& faceParam = f->getParam();

                idx.push_back(faceParam.idx[0]);
                idx.push_back(faceParam.idx[1]);
                idx.push_back(faceParam.idx[2]);
            }

            m_ib.init((uint32_t)idx.size(), &idx[0]);
        }
    }

    void objshape::addFace(face* f)
    {
        const auto& faceParam = f->getParam();

        int idx0 = faceParam.idx[0];
        int idx1 = faceParam.idx[1];
        int idx2 = faceParam.idx[2];

        int baseIdx = std::min(idx0, std::min(idx1, idx2));
        m_baseIdx = std::min(baseIdx, m_baseIdx);

        faces.push_back(f);

        m_baseTriIdx = std::min(f->getId(), m_baseTriIdx);
    }

    void objshape::drawForGBuffer(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int parentId)
    {
        if (func) {
            func(mtxL2W, mtxPrevL2W, parentId, m_baseTriIdx);
        }

        const auto& vb = ctxt.getVB();

        auto triNum = (uint32_t)faces.size();

        m_ib.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }

    void objshape::draw(
        AT_NAME::FuncObjectMeshDraw func,
        const context& ctxt)
    {
        if (func) {
            int albedoTexId = m_mtrl ? m_mtrl->param().albedoMap : -1;
            const aten::texture* albedo = albedoTexId >= 0 ? ctxt.getTexture(albedoTexId) : nullptr;

            auto color = m_mtrl ? m_mtrl->param().baseColor : vec3(1);

            auto mtrlid = m_mtrl ? m_mtrl->id() : -1;

            func(color, albedo, mtrlid);
        }

        const auto& vb = ctxt.getVB();

        auto triNum = (uint32_t)faces.size();

        m_ib.draw(vb, aten::Primitive::Triangles, 0, triNum);
    }
}
