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

        m_param.triangle_id = m_shapes[0]->triangles_[0]->GetId();

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

        m_param.triangle_id = m_shapes[0]->triangles_[0]->GetId();

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
        float t_min, float t_max,
        aten::Intersection& isect) const
    {
        bool isHit = m_accel->HitWithLod(ctxt, r, t_min, t_max, false, isect);

        if (isHit) {
            auto f = ctxt.GetTriangleInstance(isect.objid);

            // 自身のIDを返す.
            isect.objid = id();
        }
        return isHit;
    }

    void PolygonObject::render(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtx_L2W,
        const aten::mat4& mtx_prev_L2W,
        int32_t parentId,
        uint32_t triOffset)
    {
        // TODO
        // Currently ignore "triOffset"...

        int32_t objid = (parentId < 0 ? id() : parentId);

        for (auto& s : m_shapes) {
            s->render(func, ctxt, mtx_L2W, mtx_prev_L2W, objid);
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
        const aten::mat4& mtx_L2W)
    {
        m_accel->drawAABB(func, mtx_L2W);
    }

    bool PolygonObject::exportInternalAccelTree(
        const context& ctxt,
        std::string_view path)
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

    bool PolygonObject::importInternalAccelTree(
        std::string_view path, const context& ctxt, int32_t offsetTriIdx)
    {
        AT_ASSERT(!m_accel);

        m_accel = aten::accelerator::createAccelerator();
        return m_accel->importTree(ctxt, path, offsetTriIdx);
    }

    void PolygonObject::collectTriangles(std::vector<aten::TriangleParameter>& triangles) const
    {
        for (const auto& triangle_group_mesh : m_shapes) {
            const auto& tris = triangle_group_mesh->GetTriangleList();

            triangles.reserve(tris.size());

            for (const auto tri : tris) {
                triangles.push_back(tri->GetParam());
            }
        }
    }
}
