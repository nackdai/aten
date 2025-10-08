#include <random>
#include <vector>

#include "accelerator/stackless_bvh.h"
#include "accelerator/bvh.h"
#include "accelerator/bvh_util.h"
#include "geometry/transformable.h"
#include "geometry/PolygonObject.h"

//#pragma optimize( "", off)

// NOTE
// Stackless Multi-BVH Traversal for CPU, MIC and GPU Ray Tracing
// http://cg.iit.bme.hu/~afra/publications/afra2013cgf_mbvhsl.pdf

namespace aten {
    void StacklessBVH::build(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox/*= nullptr*/)
    {
        bvh_.build(ctxt, list, num, bbox);

        setBoundingBox(bvh_.GetBoundingbox());

        std::vector<accelerator*> listBvh;
        std::map<hitable*, accelerator*> nestedBvhMap;

        std::vector<std::vector<StacklessBvhNodeEntry>> listBvhNode;

        // Register to linear list to traverse bvhnode easily.
        auto root = bvh_.GetRoot();
        listBvhNode.push_back(std::vector<StacklessBvhNodeEntry>());
        registerBvhNodeToLinearList(root, nullptr, nullptr, aten::mat4::Identity, listBvhNode[0], listBvh, nestedBvhMap);

        for (int32_t i = 0; i < listBvh.size(); i++) {
            // TODO
            auto bvh = (aten::bvh*)listBvh[i];

            root = bvh->GetRoot();

            hitable* parent = nullptr;

            // Find parent which has specified bvh.
            for (auto it : nestedBvhMap) {
                if (bvh == it.second) {
                    // Found nested bvh.
                    parent = it.first;
                    break;
                }
            }

            listBvhNode.push_back(std::vector<StacklessBvhNodeEntry>());
            std::vector<accelerator*> dummy;

            registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1], dummy, nestedBvhMap);
            AT_ASSERT(dummy.empty());
        }

        m_listStacklessBvhNode.resize(listBvhNode.size());

        // Register bvh node for gpu.
        for (int32_t i = 0; i < listBvhNode.size(); i++) {
            // Leaves of nested bvh are primitive.
            // Index 0 is primiary tree, and Index N (N > 0) is nested tree.
            bool isPrimitiveLeaf = (i > 0);

            registerThreadedBvhNode(ctxt, isPrimitiveLeaf, listBvhNode[i], m_listStacklessBvhNode[i]);
        }
    }

    void StacklessBVH::registerBvhNodeToLinearList(
        bvhnode* root,
        bvhnode* parentNode,
        hitable* nestParent,
        const aten::mat4& mtx_L2W,
        std::vector<StacklessBvhNodeEntry>& listBvhNode,
        std::vector<accelerator*>& listBvh,
        std::map<hitable*, accelerator*>& nestedBvhMap)
    {
        registerBvhNodeToLinearListRecursively<StacklessBvhNodeEntry>(
            root,
            parentNode,
            nestParent,
            mtx_L2W,
            listBvhNode,
            listBvh,
            nestedBvhMap,
            [this](std::vector<StacklessBvhNodeEntry>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
        {
            list.push_back(StacklessBvhNodeEntry(node, obj, mtx));
        },
            [this](bvhnode* node, int32_t exid, int32_t subExid)
        {
            if (node->isLeaf()) {
                // NOTE
                // 0 はベースツリーなので、+1 する.
                node->setExternalId(exid);
            }
        });
    }

    void StacklessBVH::registerThreadedBvhNode(
        const context& ctxt,
        bool isPrimitiveLeaf,
        const std::vector<StacklessBvhNodeEntry>& listBvhNode,
        std::vector<StacklessBvhNode>& listStacklessBvhNode)
    {
        listStacklessBvhNode.reserve(listBvhNode.size());

        for (const auto& entry : listBvhNode) {
            auto node = entry.node;
            auto nestParent = entry.nestParent;

            StacklessBvhNode stacklessBvhNode;

            // NOTE
            // Differ set hit/miss index.

            auto bbox = node->GetBoundingbox();
            bbox = aten::aabb::transform(bbox, entry.mtx_L2W);

            // Parent id.
            auto parent = node->getParent();
            int32_t parentId = parent ? parent->getTraversalOrder() : -1;
            stacklessBvhNode.parent = (float)parentId;

            // Sibling id.
            if (parent) {
                auto left = parent->getLeft();
                auto right = parent->getRight();

                bvhnode* sibling = nullptr;

                if (left == node) {
                    sibling = right;
                }
                else if (right == node) {
                    sibling = left;
                }

                stacklessBvhNode.sibling = (float)(sibling ? sibling->getTraversalOrder() : -1);
            }
            else {
                stacklessBvhNode.sibling = -1;
            }

            if (node->isLeaf()) {
                hitable* item = node->getItem();

                // 自分自身のIDを取得.
                stacklessBvhNode.object_id = (float)ctxt.FindTransformableIdxFromPointer(item);

                // もしなかったら、ネストしているので親のIDを取得.
                if (stacklessBvhNode.object_id < 0) {
                    if (nestParent) {
                        stacklessBvhNode.object_id = (float)ctxt.FindTransformableIdxFromPointer(nestParent);
                    }
                }

                // インスタンスの実体を取得.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                stacklessBvhNode.meshid = (float)item->GetMeshId();

                if (isPrimitiveLeaf) {
                    // Leaves of this tree are primitive.
                    stacklessBvhNode.primid = (float)ctxt.FindTriangleIdxFromPointer(item);
                    stacklessBvhNode.exid = -1.0f;
                }
                else {
                    stacklessBvhNode.exid = (float)node->getExternalId();
                }

                stacklessBvhNode.boxmax_0 = aten::vec4(bbox.maxPos(), 0);
                stacklessBvhNode.boxmin_0 = aten::vec4(bbox.minPos(), 0);
            }
            else {
                auto left = node->getLeft();
                auto right = node->getRight();

                stacklessBvhNode.boxmax_0 = left ? aten::vec4(left->GetBoundingbox().maxPos(), 0) : aten::vec4(0);
                stacklessBvhNode.boxmin_0 = left ? aten::vec4(left->GetBoundingbox().minPos(), 0) : aten::vec4(0);

                stacklessBvhNode.boxmax_1 = right ? aten::vec4(right->GetBoundingbox().maxPos(), 0) : aten::vec4(0);
                stacklessBvhNode.boxmin_1 = right ? aten::vec4(right->GetBoundingbox().minPos(), 0) : aten::vec4(0);

                stacklessBvhNode.child_0 = (float)(left ? left->getTraversalOrder() : -1);
                stacklessBvhNode.child_1 = (float)(right ? right->getTraversalOrder() : -1);
            }

            listStacklessBvhNode.push_back(stacklessBvhNode);
        }
    }

    bool StacklessBVH::hit(
        const context& ctxt,
        const ray& r,
        float t_min, float t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listStacklessBvhNode, r, t_min, t_max, isect);
    }

    bool StacklessBVH::hit(
        const context& ctxt,
        int32_t exid,
        const std::vector<std::vector<StacklessBvhNode>>& listStacklessBvhNode,
        const ray& r,
        float t_min, float t_max,
        Intersection& isect,
        aten::HitStopType hit_stop_type/*= aten::HitStopType::Closest*/) const
    {
        float hitt = AT_MATH_INF;

        int32_t nodeid = 0;
        uint32_t bitstack = 0;

        for (;;) {
            const StacklessBvhNode* node = nullptr;

            if (nodeid >= 0) {
                node = &listStacklessBvhNode[exid][nodeid];
            }

            if (!node) {
                break;
            }

            bool is_hit = false;

            if (node->isLeaf()) {
                Intersection isectTmp;

                auto s = ctxt.GetTransformable((int32_t)node->object_id);

                if (node->exid >= 0) {
                    // Traverse external linear bvh list.
                    const auto& param = s->GetParam();

                    int32_t mtx_id = param.mtx_id;

                    aten::ray transformedRay;

                    if (mtx_id >= 0) {
                        const auto& mtx_W2L = ctxt.GetMatrix(mtx_id);

                        transformedRay = mtx_W2L.applyRay(r);
                    }
                    else {
                        transformedRay = r;
                    }

                    is_hit = hit(
                        ctxt,
                        (int32_t)node->exid,
                        listStacklessBvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else if (node->primid >= 0) {
                    // Hit test for a primitive.
                    auto prim = ctxt.GetTriangleInstance((int32_t)node->primid);
                    is_hit = prim->hit(ctxt, r, t_min, t_max, isectTmp);
                    if (is_hit) {
                        isectTmp.objid = s->id();
                    }
                }
                else {
                    // Hit test for a shape.
                    is_hit = s->hit(ctxt, r, t_min, t_max, isectTmp);
                }

                if (is_hit) {
                    float tmp_t_max = hit_stop_type == aten::HitStopType::Any
                        ? AT_MATH_INF
                        : t_max;

                    if (isectTmp.t < tmp_t_max) {
                        isect = isectTmp;
                        t_max = isect.t;

                        if (hit_stop_type == aten::HitStopType::Any
                            || hit_stop_type == aten::HitStopType::Closer)
                        {
                            break;
                        }
                    }
                }
            }
            else {
                float t[2];
                bool hit[2];

                hit[0] = aten::aabb::hit(r, node->boxmin_0, node->boxmax_0, t_min, t_max, &t[0]);
                hit[1] = aten::aabb::hit(r, node->boxmin_1, node->boxmax_1, t_min, t_max, &t[1]);

                if (hit[0] || hit[1]) {
                    bitstack = bitstack << 1;

                    if (hit[0] && hit[1]) {
                        nodeid = (int32_t)(t[0] < t[1] ? node->child_0 : node->child_1);
                        bitstack = bitstack | 1;
                    }
                    else if (hit[0]) {
                        nodeid = (int32_t)node->child_0;
                    }
                    else if (hit[1]) {
                        nodeid = (int32_t)node->child_1;
                    }

                    continue;
                }
            }

            while ((bitstack & 1) == 0) {
                if (bitstack == 0) {
                    return (isect.objid >= 0);
                }

                nodeid = (int32_t)node->parent;
                bitstack = bitstack >> 1;

                node = &listStacklessBvhNode[exid][nodeid];
            }

            nodeid = (int32_t)node->sibling;
            bitstack = bitstack ^ 1;
        }

        return (isect.objid >= 0);
    }
}
