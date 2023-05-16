#include <iterator>

#include "geometry/triangle.h"
#include "geometry/TriangleGroupMesh.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
    std::shared_ptr<triangle> triangle::create(
        const context& ctxt,
        const aten::TriangleParameter& param_)
    {
        auto f = std::make_shared<triangle>();

        f->param_ = param_;

        f->build(ctxt, param_.mtrlid, param_.mesh_id);

        return f;
    }

    bool triangle::hit(
        const context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        const auto& v0 = ctxt.getVertex(param_.idx[0]);
        const auto& v1 = ctxt.getVertex(param_.idx[1]);
        const auto& v2 = ctxt.getVertex(param_.idx[2]);

        bool isHit = hit(
            &param_,
            v0.pos, v1.pos, v2.pos,
            r,
            t_min, t_max,
            &isect);

        if (isHit) {
            // Temporary, notify triangle id to the parent object.
            isect.objid = m_id;

            isect.triangle_id = m_id;

            isect.mtrlid = param_.mtrlid;
        }

        return isHit;
    }

    bool triangle::hit(
        const aten::TriangleParameter* param_,
        const aten::vec3& v0,
        const aten::vec3& v1,
        const aten::vec3& v2,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection* isect)
    {
        bool isHit = false;

        const auto res = intersectTriangle(r, v0, v1, v2);

        if (res.isIntersect) {
            if (res.t < isect->t) {
                isect->t = res.t;

                isect->a = res.a;
                isect->b = res.b;

                isHit = true;
            }
        }

        return isHit;
    }

    void triangle::evalHitResult(
        const aten::context& ctxt,
        const aten::TriangleParameter& tri,
        aten::hitrecord* rec,
        const aten::TriangleParameter& param,
        const aten::Intersection* isect)
    {
        const auto p0 = ctxt.GetPositionAsVec4(tri.idx[0]);
        const auto p1 = ctxt.GetPositionAsVec4(tri.idx[1]);
        const auto p2 = ctxt.GetPositionAsVec4(tri.idx[2]);

        const auto n0 = ctxt.GetNormalAsVec4(tri.idx[0]);
        const auto n1 = ctxt.GetNormalAsVec4(tri.idx[1]);
        const auto n2 = ctxt.GetNormalAsVec4(tri.idx[2]);

        const auto u0 = p0.w;
        const auto v0 = n0.w;

        const auto u1 = p1.w;
        const auto v1 = n1.w;

        const auto u2 = p2.w;
        const auto v2 = n2.w;

        // NOTE
        // http://d.hatena.ne.jp/Zellij/20131207/p1

        real a = isect->a;
        real b = isect->b;
        real c = 1 - a - b;

        // 重心座標系(barycentric coordinates).
        // v0基準.
        // p = (1 - a - b)*v0 + a*v1 + b*v2
        rec->p = c * p0 + a * p1 + b * p2;
        rec->normal = c * n0 + a * n1 + b * n2;

        rec->u = c * u0 + a * u1 + b * u2;
        rec->v = c * v0 + a * v1 + b * v2;

        if (param.needNormal > 0) {
            auto e01 = p1 - p0;
            auto e02 = p2 - p0;

            e01.w = e02.w = real(0);

            rec->normal = normalize(cross(e01, e02));
        }

        rec->area = param.area;
    }

    void triangle::build(
        const context& ctxt,
        int32_t mtrlid,
        int32_t geomid)
    {
        const auto& v0 = ctxt.getVertex(param_.idx[0]);
        const auto& v1 = ctxt.getVertex(param_.idx[1]);
        const auto& v2 = ctxt.getVertex(param_.idx[2]);

        aten::vec3 vmax = aten::vec3(
            std::max(v0.pos.x, std::max(v1.pos.x, v2.pos.x)),
            std::max(v0.pos.y, std::max(v1.pos.y, v2.pos.y)),
            std::max(v0.pos.z, std::max(v1.pos.z, v2.pos.z)));

        aten::vec3 vmin = aten::vec3(
            std::min(v0.pos.x, std::min(v1.pos.x, v2.pos.x)),
            std::min(v0.pos.y, std::min(v1.pos.y, v2.pos.y)),
            std::min(v0.pos.z, std::min(v1.pos.z, v2.pos.z)));

        setBoundingBox(aten::aabb(vmin, vmax));

        // 三角形の面積 = ２辺の外積の長さ / 2;
        auto e0 = v1.pos - v0.pos;
        auto e1 = v2.pos - v0.pos;
        param_.area = real(0.5) * cross(e0, e1).length();

        param_.mtrlid = mtrlid;
        param_.mesh_id = geomid;
    }

    int32_t triangle::mesh_id() const
    {
        return param_.mesh_id;
    }

    aabb triangle::computeAABB(const context& ctxt) const
    {
        const auto& v0 = ctxt.getVertex(param_.idx[0]);
        const auto& v1 = ctxt.getVertex(param_.idx[1]);
        const auto& v2 = ctxt.getVertex(param_.idx[2]);

        auto vmin = aten::min(aten::min(v0.pos, v1.pos), v2.pos);
        auto vmax = aten::max(aten::max(v0.pos, v1.pos), v2.pos);

        aabb ret(vmin, vmax);

        return ret;
    }
}
