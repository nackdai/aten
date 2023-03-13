#include <iterator>

#include "geometry/PolygonObject.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
    void PolygonObject::build(const context& ctxt)
    {
        if (m_param.triangle_num > 0) {
            // Builded already.
            return;
        }

        if (!m_accel) {
            m_accel = aten::accelerator::createAccelerator();
        }

        m_param.triangle_id = m_shapes[0]->triangles_[0]->getId();

        m_param.area = 0;
        uint32_t triangles = 0;

        // Avoid sorting triangle group mesh list in bvh::build directly.
        std::vector<triangle*> tmp;

        aabb bbox;

        for (const auto& s : m_shapes) {
            auto mesh_area = s->build(ctxt);

            m_param.area += mesh_area;
            triangles += (uint32_t)s->triangles_.size();

            for (const auto f : s->triangles_) {
                tmp.push_back(f.get());
            }

            aabb::merge(bbox, s->m_aabb);
        }

        m_param.triangle_num = triangles;

        m_accel->asNested();
        m_accel->build(ctxt, (hitable**)&tmp[0], (uint32_t)tmp.size(), &bbox);

        bbox = m_accel->getBoundingbox();

        setBoundingBox(bbox);
    }

    void PolygonObject::buildForRasterizeRendering(const context& ctxt)
    {
        if (m_param.triangle_num > 0) {
            // Builded already.
            return;
        }

        m_param.triangle_id = m_shapes[0]->triangles_[0]->getId();

        m_param.area = 0;
        uint32_t triangles = 0;

        for (const auto& s : m_shapes) {
            s->build(ctxt);

            triangles += (uint32_t)s->triangles_.size();
        }

        m_param.triangle_num = triangles;
    }

    bool PolygonObject::hit(
        const context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        bool isHit = m_accel->hit(ctxt, r, t_min, t_max, false, isect);

        if (isHit) {
            auto f = ctxt.getTriangle(isect.objid);

            // 自身のIDを返す.
            isect.objid = id();
        }
        return isHit;
    }

    void PolygonObject::evalHitResult(
        const context& ctxt,
        const aten::ray& r,
        const aten::mat4& mtxL2W,
        aten::hitrecord& rec,
        const aten::Intersection& isect) const
    {
        auto f = ctxt.getTriangle(isect.triangle_id);

        auto& vtxs = ctxt.getVertices();

        const auto& faceParam = f->getParam();

        const auto& v0 = vtxs[faceParam.idx[0]];
        const auto& v1 = vtxs[faceParam.idx[1]];
        const auto& v2 = vtxs[faceParam.idx[2]];

        //triangle::evalHitResult(v0, v1, v2, &rec, &isect);
        f->evalHitResult(ctxt, r, rec, isect);

        real orignalLen = 0;
        {
            const auto& p0 = v0.pos;
            const auto& p1 = v1.pos;

            orignalLen = length(p1.v - p0.v);
        }

        real scaledLen = 0;
        {
            auto p0 = mtxL2W.apply(v0.pos);
            auto p1 = mtxL2W.apply(v1.pos);

            scaledLen = length(p1.v - p0.v);
        }

        real ratio = scaledLen / orignalLen;
        ratio = ratio * ratio;

        rec.area = m_param.area * ratio;

        rec.mtrlid = isect.mtrlid;
    }

    void PolygonObject::getSamplePosNormalArea(
        const context& ctxt,
        aten::SamplePosNormalPdfResult* result,
        const aten::mat4& mtxL2W,
        aten::sampler* sampler) const
    {
        auto r = sampler->nextSample();
        int32_t shapeidx = (int32_t)(r * (m_shapes.size() - 1));
        auto& triangle_group_mesh = m_shapes[shapeidx];

        r = sampler->nextSample();
        int32_t faceidx = (int32_t)(r * (triangle_group_mesh->triangles_.size() - 1));
        auto f = triangle_group_mesh->triangles_[faceidx];

        const auto& faceParam = f->getParam();

        const auto& v0 = ctxt.getVertex(faceParam.idx[0]);
        const auto& v1 = ctxt.getVertex(faceParam.idx[1]);

        real orignalLen = 0;
        {
            const auto& p0 = v0.pos;
            const auto& p1 = v1.pos;

            orignalLen = (p1 - p0).length();
        }

        real scaledLen = 0;
        {
            auto p0 = mtxL2W.apply(v0.pos);
            auto p1 = mtxL2W.apply(v1.pos);

            scaledLen = length(p1.v - p0.v);
        }

        real ratio = scaledLen / orignalLen;
        ratio = ratio * ratio;

        auto area = m_param.area * ratio;

        f->getSamplePosNormalArea(ctxt, result, sampler);

        result->area = area;
    }

    void PolygonObject::render(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int32_t parentId,
        uint32_t triOffset)
    {
        // TODO
        // Currently ignore "triOffset"...

        int32_t objid = (parentId < 0 ? id() : parentId);

        for (auto& s : m_shapes) {
            s->render(func, ctxt, mtxL2W, mtxPrevL2W, objid);
        }
    }

    void PolygonObject::draw(
        AT_NAME::FuncObjectMeshDraw func,
        const context& ctxt) const
    {
        for (auto& s : m_shapes) {
            s->draw(func, ctxt);
        }
    }

    void PolygonObject::drawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W)
    {
        m_accel->drawAABB(func, mtxL2W);
    }

    bool PolygonObject::exportInternalAccelTree(
        const context& ctxt,
        const char* path)
    {
        bool result = false;

        m_accel = aten::accelerator::createAccelerator();
        m_accel->enableExporting();

        build(ctxt);

        if (m_accel) {
            result = m_accel->exportTree(ctxt, path);
        }

        return result;
    }

    bool PolygonObject::importInternalAccelTree(const char* path, const context& ctxt, int32_t offsetTriIdx)
    {
        AT_ASSERT(!m_accel);

        m_accel = aten::accelerator::createAccelerator();
        return m_accel->importTree(ctxt, path, offsetTriIdx);
    }

    void PolygonObject::collectTriangles(std::vector<aten::TriangleParameter>& triangles) const
    {
        for (const auto& triangle_group_mesh : m_shapes) {
            const auto& tris = triangle_group_mesh->tris();

            triangles.reserve(tris.size());

            for (const auto tri : tris) {
                triangles.push_back(tri->getParam());
            }
        }
    }
}
