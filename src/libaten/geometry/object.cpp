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

    void object::build()
    {
        if (m_triangles > 0) {
            // Builded already.
            return;
        }

        if (!m_accel) {
            m_accel = aten::accelerator::createAccelerator();
        }

        param.primid = shapes[0]->faces[0]->id;

        param.area = 0;
        m_triangles = 0;

        // Avoid sorting objshape list in bvh::build directly.
        std::vector<face*> tmp;

        aabb bbox;

        for (const auto s : shapes) {
            s->build();

            param.area += s->param.area;
            m_triangles += (uint32_t)s->faces.size();

            tmp.insert(tmp.end(), s->faces.begin(), s->faces.end());

            aabb::merge(bbox, s->m_aabb);
        }

        param.primnum = m_triangles;

        m_accel->asNested();
        m_accel->build((hitable**)&tmp[0], (uint32_t)tmp.size(), &bbox);

        bbox = m_accel->getBoundingbox();

        setBoundingBox(bbox);
    }

    void object::buildForRasterizeRendering()
    {
        if (m_triangles > 0) {
            // Builded already.
            return;
        }

        param.primid = shapes[0]->faces[0]->id;

        param.area = 0;
        m_triangles = 0;

        for (const auto s : shapes) {
            s->build();

            m_triangles += (uint32_t)s->faces.size();
        }

        param.primnum = m_triangles;
    }

    bool object::hit(
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        bool isHit = m_accel->hit(r, t_min, t_max, isect, false);

        if (isHit) {
            auto f = face::faces()[isect.objid];

            // Ž©g‚ÌID‚ð•Ô‚·.
            isect.objid = id();
        }
        return isHit;
    }

    void object::evalHitResult(
        const aten::ray& r,
        const aten::mat4& mtxL2W,
        aten::hitrecord& rec,
        const aten::Intersection& isect) const
    {
        auto f = face::faces()[isect.primid];

        auto& vtxs = aten::VertexManager::getVertices();

        const auto& v0 = vtxs[f->param.idx[0]];
        const auto& v1 = vtxs[f->param.idx[1]];
        const auto& v2 = vtxs[f->param.idx[2]];

        //face::evalHitResult(v0, v1, v2, &rec, &isect);
        f->evalHitResult(r, rec, isect);

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

        rec.area = param.area * ratio;

        rec.mtrlid = isect.mtrlid;
    }

    void object::getSamplePosNormalArea(
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

        const auto& v0 = aten::VertexManager::getVertex(f->param.idx[0]);
        const auto& v1 = aten::VertexManager::getVertex(f->param.idx[1]);

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

        auto area = param.area * ratio;

        f->getSamplePosNormalArea(result, sampler);

        result->area = area;
    }

    void object::draw(
        aten::hitable::FuncPreDraw func,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int parentId,
        uint32_t triOffset)
    {
        // TODO
        // Currently ignore "triOffset"...

        int objid = (parentId < 0 ? id() : parentId);

        for (auto s : shapes) {
            s->draw(func, mtxL2W, mtxPrevL2W, objid);
        }
    }

    void object::draw(AT_NAME::FuncObjectMeshDraw func)
    {
        for (auto s : shapes) {
            s->draw(func);
        }
    }

    void object::drawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W)
    {
        m_accel->drawAABB(func, mtxL2W);
    }

    bool object::exportInternalAccelTree(const char* path)
    {
        bool result = false;

        m_accel = aten::accelerator::createAccelerator();
        m_accel->enableExporting();

        build();

        if (m_accel) {
            result = m_accel->exportTree(path);
        }

        return result;
    }

    bool object::importInternalAccelTree(const char* path, int offsetTriIdx/*= 0*/)
    {
        AT_ASSERT(!m_accel);

        m_accel = aten::accelerator::createAccelerator();
        return m_accel->importTree(path, offsetTriIdx);
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
                triangles.push_back(tri->param);
            }
        }
    }
}
