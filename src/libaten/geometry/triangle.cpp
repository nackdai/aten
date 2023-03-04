#include <iterator>

#include "geometry/triangle.h"
#include "geometry/objshape.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
    std::shared_ptr<triangle> triangle::create(
        const context& ctxt,
        const aten::TriangleParameter& param)
    {
        auto f = std::make_shared<triangle>();

        f->param = param;

        f->build(ctxt, param.mtrlid, param.mesh_id);

        return f;
    }

    bool triangle::hit(
        const context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const
    {
        const auto& v0 = ctxt.getVertex(param.idx[0]);
        const auto& v1 = ctxt.getVertex(param.idx[1]);
        const auto& v2 = ctxt.getVertex(param.idx[2]);

        bool isHit = hit(
            &param,
            v0.pos, v1.pos, v2.pos,
            r,
            t_min, t_max,
            &isect);

        if (isHit) {
            // Temporary, notify triangle id to the parent object.
            isect.objid = m_id;

            isect.triangle_id = m_id;

            isect.mtrlid = param.mtrlid;
        }

        return isHit;
    }

    bool triangle::hit(
        const aten::TriangleParameter* param,
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
        const context& ctxt,
        const aten::ray& r,
        aten::hitrecord& rec,
        const aten::Intersection& isect) const
    {
        const auto& v0 = ctxt.getVertex(param.idx[0]);
        const auto& v1 = ctxt.getVertex(param.idx[1]);
        const auto& v2 = ctxt.getVertex(param.idx[2]);

        evalHitResult(v0, v1, v2, &rec, &isect);

        if (param.needNormal > 0) {
            auto e01 = v1.pos - v0.pos;
            auto e02 = v2.pos - v0.pos;

            e01.w = e02.w = real(0);

            rec.normal = normalize(cross(e01, e02));
        }

        rec.area = param.area;
    }

    void triangle::evalHitResult(
        const aten::vertex& v0,
        const aten::vertex& v1,
        const aten::vertex& v2,
        aten::hitrecord* rec,
        const aten::Intersection* isect)
    {
        // NOTE
        // http://d.hatena.ne.jp/Zellij/20131207/p1

        real a = isect->a;
        real b = isect->b;
        real c = 1 - a - b;

        // 重心座標系(barycentric coordinates).
        // v0基準.
        // p = (1 - a - b)*v0 + a*v1 + b*v2
        rec->p = c * v0.pos + a * v1.pos + b * v2.pos;
        rec->normal = c * v0.nml + a * v1.nml + b * v2.nml;
        auto uv = c * v0.uv + a * v1.uv + b * v2.uv;

        rec->u = uv.x;
        rec->v = uv.y;
    }

    void triangle::build(
        const context& ctxt,
        int32_t mtrlid,
        int32_t geomid)
    {
        const auto& v0 = ctxt.getVertex(param.idx[0]);
        const auto& v1 = ctxt.getVertex(param.idx[1]);
        const auto& v2 = ctxt.getVertex(param.idx[2]);

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
        param.area = real(0.5) * cross(e0, e1).length();

        param.mtrlid = mtrlid;
        param.mesh_id = geomid;
    }

    void triangle::getSamplePosNormalArea(
        const context& ctxt,
        aten::SamplePosNormalPdfResult* result,
        aten::sampler* sampler) const
    {
#if 0
        // 0 <= a + b <= 1
        real a = sampler->nextSample();
        real b = sampler->nextSample();

        real d = a + b;

        if (d > 1) {
            a /= d;
            b /= d;
        }
#else
        real r0 = sampler->nextSample();
        real r1 = sampler->nextSample();

        real a = aten::sqrt(r0) * (real(1) - r1);
        real b = aten::sqrt(r0) * r1;
#endif

        const auto& v0 = ctxt.getVertex(param.idx[0]);
        const auto& v1 = ctxt.getVertex(param.idx[1]);
        const auto& v2 = ctxt.getVertex(param.idx[2]);

        // 重心座標系(barycentric coordinates).
        // v0基準.
        // p = (1 - a - b)*v0 + a*v1 + b*v2
        aten::vec3 p = (1 - a - b) * v0.pos + a * v1.pos + b * v2.pos;

        aten::vec3 n = (1 - a - b) * v0.nml + a * v1.nml + b * v2.nml;
        n = normalize(n);

        // 三角形の面積 = ２辺の外積の長さ / 2;
        auto e0 = v1.pos - v0.pos;
        auto e1 = v2.pos - v0.pos;
        auto area = real(0.5) * cross(e0, e1).length();

        result->pos = p;
        result->nml = n;
        result->area = area;

        result->a = a;
        result->b = b;

        result->triangle_id = m_id;
    }

    int32_t triangle::geomid() const
    {
        return param.mesh_id;
    }

    aabb triangle::computeAABB(const context& ctxt) const
    {
        const auto& v0 = ctxt.getVertex(param.idx[0]);
        const auto& v1 = ctxt.getVertex(param.idx[1]);
        const auto& v2 = ctxt.getVertex(param.idx[2]);

        auto vmin = aten::min(aten::min(v0.pos, v1.pos), v2.pos);
        auto vmax = aten::max(aten::max(v0.pos, v1.pos), v2.pos);

        aabb ret(vmin, vmax);

        return ret;
    }
}
