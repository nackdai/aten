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
        float t_min, float t_max,
        aten::Intersection& isect) const
    {
        bool isHit = hit(
            param_,
            ctxt,
            r,
            &isect);

        if (isHit) {
            // Temporary, notify triangle id to the parent object.
            isect.objid = m_id;

            isect.triangle_id = m_id;

            isect.mtrlid = param_.mtrlid;
        }

        return isHit;
    }

    void triangle::build(
        const context& ctxt,
        int32_t mtrlid,
        int32_t geomid)
    {
        const auto& v0 = ctxt.GetVertex(param_.idx[0]);
        const auto& v1 = ctxt.GetVertex(param_.idx[1]);
        const auto& v2 = ctxt.GetVertex(param_.idx[2]);

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
        param_.area = float(0.5) * cross(e0, e1).length();

        param_.mtrlid = mtrlid;
        param_.mesh_id = geomid;
    }

    int32_t triangle::GetMeshId() const
    {
        return param_.mesh_id;
    }

    aabb triangle::ComputeAABB(const context& ctxt) const
    {
        const auto& v0 = ctxt.GetVertex(param_.idx[0]);
        const auto& v1 = ctxt.GetVertex(param_.idx[1]);
        const auto& v2 = ctxt.GetVertex(param_.idx[2]);

        auto vmin = aten::vmin(aten::vmin(v0.pos, v1.pos), v2.pos);
        auto vmax = aten::vmax(aten::vmax(v0.pos, v1.pos), v2.pos);

        aabb ret(vmin, vmax);

        return ret;
    }
}
