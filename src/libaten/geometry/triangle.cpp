#include <iterator>

#include "geometry/triangle.h"
#include "geometry/TriangleGroupMesh.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

//#define ENABLE_LINEAR_HITTEST

namespace aten
{
    std::shared_ptr<triangle> triangle::create(
        aten::context& ctxt,
        const aten::TriangleParameter& param_,
        const std::optional<aten::vec3>& scale)
    {
        auto f = std::make_shared<triangle>();

        f->param_ = param_;

        f->build(ctxt, param_.v1.mtrlid, param_.v1.mesh_id, scale);

        return f;
    }

    bool triangle::hit(
        const aten::context& ctxt,
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

            isect.mtrlid = param_.v1.mtrlid;
        }

        return isHit;
    }

    void triangle::build(
        aten::context& ctxt,
        int32_t mtrlid,
        int32_t geomid,
        const std::optional<aten::vec3>& scale)
    {
        if (scale.has_value()) {
            auto v0 = ctxt.GetVertex(param_.v0.idx[0]);
            auto v1 = ctxt.GetVertex(param_.v0.idx[1]);
            auto v2 = ctxt.GetVertex(param_.v0.idx[2]);

            const auto& real_scale = scale.value();

            v0.pos *= real_scale;
            v1.pos *= real_scale;
            v2.pos *= real_scale;

            BuildTriangle(
                ctxt,
                v0, v1, v2,
                mtrlid, geomid);

#if 0
            // Check if the normal needs to be computed on the fly while traversing bvh.
            const bool need_compute_normal_on_the_fly = param_.needNormal;

            // If no, normal won't be computed while rendering.
            // In that case, normal needs to be here.
            if (!need_compute_normal_on_the_fly) {
                auto e0 = v1.pos - v0.pos;
                auto e1 = v2.pos - v0.pos;
                v0.nml = normalize(cross(e0, e1));

                e0 = v2.pos - v1.pos;
                e1 = v0.pos - v1.pos;
                v1.nml = normalize(cross(e0, e1));

                e0 = v0.pos - v2.pos;
                e1 = v1.pos - v2.pos;
                v2.nml = normalize(cross(e0, e1));
            }
#endif

            ctxt.ReplaceVertex(param_.v0.idx[0], v0);
            ctxt.ReplaceVertex(param_.v0.idx[1], v1);
            ctxt.ReplaceVertex(param_.v0.idx[2], v2);
        }
        else {
            const auto& v0 = ctxt.GetVertex(param_.v0.idx[0]);
            const auto& v1 = ctxt.GetVertex(param_.v0.idx[1]);
            const auto& v2 = ctxt.GetVertex(param_.v0.idx[2]);
            BuildTriangle(
                ctxt,
                v0, v1, v2,
                mtrlid, geomid);
        }
    }

    void triangle::BuildTriangle(
        const aten::context& ctxt,
        const aten::vertex& v0,
        const aten::vertex& v1,
        const aten::vertex& v2,
        int32_t mtrlid,
        int32_t geomid)
    {
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
        param_.v1.area = float(0.5) * cross(e0, e1).length();

        param_.v1.mtrlid = mtrlid;
        param_.v1.mesh_id = geomid;
    }

    int32_t triangle::GetMeshId() const
    {
        return param_.v1.mesh_id;
    }

    aabb triangle::ComputeAABB(const aten::context& ctxt) const
    {
        const auto& v0 = ctxt.GetVertex(param_.v0.idx[0]);
        const auto& v1 = ctxt.GetVertex(param_.v0.idx[1]);
        const auto& v2 = ctxt.GetVertex(param_.v0.idx[2]);

        auto vmin = aten::vmin(aten::vmin(v0.pos, v1.pos), v2.pos);
        auto vmax = aten::vmax(aten::vmax(v0.pos, v1.pos), v2.pos);

        aabb ret(vmin, vmax);

        return ret;
    }
}
