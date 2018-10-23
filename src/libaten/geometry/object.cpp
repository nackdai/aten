#include <iterator>

#include "geometry/object.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
    object::~object()
    {
        for (auto s : shapes) {
            delete s;
        }
        shapes.clear();

        if (m_accel) {
            delete m_accel;
        }
    }

    void object::build(const context& ctxt)
    {
        if (m_triangles > 0) {
            // Builded already.
            return;
        }

        if (!m_accel) {
            m_accel = aten::accelerator::createAccelerator();
        }

        m_param.primid = shapes[0]->faces[0]->getId();

        m_param.area = 0;
        m_triangles = 0;

        // Avoid sorting objshape list in bvh::build directly.
        std::vector<face*> tmp;

        aabb bbox;

        for (const auto s : shapes) {
            s->build(ctxt);

            m_param.area += s->param.area;
            m_triangles += (uint32_t)s->faces.size();

            tmp.insert(tmp.end(), s->faces.begin(), s->faces.end());

            aabb::merge(bbox, s->m_aabb);
        }

        m_param.primnum = m_triangles;

        m_accel->asNested();
        m_accel->build(ctxt, (hitable**)&tmp[0], (uint32_t)tmp.size(), &bbox);

        bbox = m_accel->getBoundingbox();

        setBoundingBox(bbox);
    }

    void object::buildForRasterizeRendering(const context& ctxt)
    {
        if (m_triangles > 0) {
            // Builded already.
            return;
        }

        m_param.primid = shapes[0]->faces[0]->getId();

        m_param.area = 0;
        m_triangles = 0;

        for (const auto s : shapes) {
            s->build(ctxt);

            m_triangles += (uint32_t)s->faces.size();
        }

        m_param.primnum = m_triangles;
    }

    bool object::hit(
        const context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        bool isHit = m_accel->hit(ctxt, r, t_min, t_max, false, isect);

        if (isHit) {
            auto f = ctxt.getTriangle(isect.objid);

            // Ž©g‚ÌID‚ð•Ô‚·.
            isect.objid = id();
        }
        return isHit;
    }

    void object::evalHitResult(
        const context& ctxt,
        const aten::ray& r,
        const aten::mat4& mtxL2W,
        aten::hitrecord& rec,
        const aten::Intersection& isect) const
    {
        auto f = ctxt.getTriangle(isect.primid);

        auto& vtxs = ctxt.getVertices();

        const auto& faceParam = f->getParam();

        const auto& v0 = vtxs[faceParam.idx[0]];
        const auto& v1 = vtxs[faceParam.idx[1]];
        const auto& v2 = vtxs[faceParam.idx[2]];

        //face::evalHitResult(v0, v1, v2, &rec, &isect);
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

    void object::getSamplePosNormalArea(
        const context& ctxt,
        aten::hitable::SamplePosNormalPdfResult* result,
        const aten::mat4& mtxL2W, 
        aten::sampler* sampler) const
    {
        auto r = sampler->nextSample();
        int shapeidx = (int)(r * (shapes.size() - 1));
        auto objshape = shapes[shapeidx];

        r = sampler->nextSample();
        int faceidx = (int)(r * (objshape->faces.size() - 1));
        auto f = objshape->faces[faceidx];

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

    void object::drawForGBuffer(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int parentId,
        uint32_t triOffset)
    {
        // TODO
        // Currently ignore "triOffset"...

        int objid = (parentId < 0 ? id() : parentId);

        for (auto s : shapes) {
            s->drawForGBuffer(func, ctxt, mtxL2W, mtxPrevL2W, objid);
        }
    }

    void object::draw(
        AT_NAME::FuncObjectMeshDraw func,
        const context& ctxt) const
    {
        for (auto s : shapes) {
            s->draw(func, ctxt);
        }
    }

    void object::drawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W)
    {
        m_accel->drawAABB(func, mtxL2W);
    }

    bool object::exportInternalAccelTree(
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

    bool object::importInternalAccelTree(const char* path, const context& ctxt, int offsetTriIdx)
    {
        AT_ASSERT(!m_accel);

        m_accel = aten::accelerator::createAccelerator();
        return m_accel->importTree(ctxt, path, offsetTriIdx);
    }

    void object::gatherTrianglesAndMaterials(
        std::vector<std::vector<AT_NAME::face*>>& tris,
        std::vector<AT_NAME::material*>& mtrls)
    {
        tris.resize(shapes.size());

        for (int i = 0; i < shapes.size(); i++) {
            auto shape = shapes[i];

            for (auto face : shape->faces) {
                tris[i].push_back(face);
            }

            mtrls.push_back(shape->m_mtrl);
        }
    }

    void object::collectTriangles(std::vector<aten::PrimitiveParamter>& triangles) const
    {
        for (const auto objshape : shapes) {
            const auto& tris = objshape->tris();

            triangles.reserve(tris.size());

            for (const auto tri : tris) {
                triangles.push_back(tri->getParam());
            }
        }
    }
}
