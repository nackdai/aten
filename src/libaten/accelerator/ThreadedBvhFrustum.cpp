#include <random>
#include <vector>
#include <iterator>

#include "accelerator/threaded_bvh.h"
#include "accelerator/bvh.h"
#include "geometry/transformable.h"
#include "geometry/object.h"

namespace aten
{
    accelerator::ResultIntersectTestByFrustum ThreadedBVH::intersectTestByFrustum(const frustum& f)
    {
        return std::move(m_bvh.intersectTestByFrustum(f));
    }

    bool ThreadedBVH::hitMultiLevel(
        const accelerator::ResultIntersectTestByFrustum& fisect,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hitMultiLevel(
            fisect.ex, fisect.ep, fisect.top,
            m_listThreadedBvhNode,
            r,
            t_min, t_max,
            isect);
    }

    bool ThreadedBVH::hitMultiLevel(
        int exid,
        int nodeid,
        int topid,
        const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        auto& shapes = transformable::getShapes();
        auto& prims = face::faces();

        real hitt = AT_MATH_INF;

        for (;;) {
            const ThreadedBvhNode* node = nullptr;

            if (nodeid >= 0) {
                node = &listThreadedBvhNode[exid][nodeid];
            }

            if (!node) {
                break;
            }

            bool isHit = false;

            if (node->isLeaf()) {
                Intersection isectTmp;

                auto s = shapes[(int)node->shapeid];

                if (node->exid >= 0) {
                    // Traverse external linear bvh list.
                    const auto& param = s->getParam();

                    int mtxid = param.mtxid;

                    aten::ray transformedRay;

                    if (mtxid >= 0) {
                        const auto& mtxW2L = m_mtxs[mtxid * 2 + 1];

                        transformedRay = mtxW2L.applyRay(r);
                    }
                    else {
                        transformedRay = r;
                    }

                    int exid = node->mainExid;

                    isHit = hit(
                        exid,
                        listThreadedBvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else if (node->primid >= 0) {
                    // Hit test for a primitive.
                    auto prim = (hitable*)prims[(int)node->primid];
                    isHit = prim->hit(r, t_min, t_max, isectTmp);
                    if (isHit) {
                        isectTmp.objid = s->id();
                    }
                }
                else {
                    // Hit test for a shape.
                    isHit = s->hit(r, t_min, t_max, isectTmp);
                }

                if (isHit) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;
                    }
                }
            }
            else {
                isHit = aten::aabb::hit(r, node->boxmin, node->boxmax, t_min, t_max);
            }

            if (isHit) {
                nodeid = (int)node->hit;
            }
            else {
                nodeid = (int)node->miss;
            }
        }

        bool ret = (isect.objid >= 0);

        if (!ret && exid > 0) {
            ret = hitMultiLevel(
                0, topid, -1,
                listThreadedBvhNode,
                r,
                t_min, t_max,
                isect);
        }

        return ret;
    }
}
