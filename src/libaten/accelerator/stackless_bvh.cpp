#include <random>
#include <vector>

#include "accelerator/stackless_bvh.h"
#include "accelerator/bvh.h"
#include "geometry/transformable.h"
#include "geometry/object.h"

//#pragma optimize( "", off)

// NOTE
// http://cg.iit.bme.hu/~afra/publications/afra2013cgf_mbvhsl.pdf

namespace aten {
    void StacklessBVH::build(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox/*= nullptr*/)
    {
        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());

        // Gather local-world matrix.
        ctxt.copyMatricesAndUpdateTransformableMatrixIdx(m_mtxs);

        std::vector<accelerator*> listBvh;
        std::map<hitable*, accelerator*> nestedBvhMap;

        std::vector<std::vector<StacklessBvhNodeEntry>> listBvhNode;
        
        // Register to linear list to traverse bvhnode easily.
        auto root = m_bvh.getRoot();
        listBvhNode.push_back(std::vector<StacklessBvhNodeEntry>());
        registerBvhNodeToLinearList(root, nullptr, nullptr, aten::mat4::Identity, listBvhNode[0], listBvh, nestedBvhMap);

        for (int i = 0; i < listBvh.size(); i++) {
            // TODO
            auto bvh = (aten::bvh*)listBvh[i];

            root = bvh->getRoot();

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
        for (int i = 0; i < listBvhNode.size(); i++) {
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
        const aten::mat4& mtxL2W,
        std::vector<StacklessBvhNodeEntry>& listBvhNode,
        std::vector<accelerator*>& listBvh,
        std::map<hitable*, accelerator*>& nestedBvhMap)
    {
        bvh::registerBvhNodeToLinearList<StacklessBvhNodeEntry>(
            root,
            parentNode,
            nestParent,
            mtxL2W,
            listBvhNode,
            listBvh,
            nestedBvhMap,
            [this](std::vector<StacklessBvhNodeEntry>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
        {
            list.push_back(StacklessBvhNodeEntry(node, obj, mtx));
        },
            [this](bvhnode* node, int exid, int subExid)
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

            auto bbox = node->getBoundingbox();
            bbox = aten::aabb::transform(bbox, entry.mtxL2W);

            // Parent id.
            auto parent = node->getParent();
            int parentId = parent ? parent->getTraversalOrder() : -1;
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
                stacklessBvhNode.shapeid = (float)ctxt.findTransformableIdxFromPointer(item);

                // もしなかったら、ネストしているので親のIDを取得.
                if (stacklessBvhNode.shapeid < 0) {
                    if (nestParent) {
                        stacklessBvhNode.shapeid = (float)ctxt.findTransformableIdxFromPointer(nestParent);
                    }
                }

                // インスタンスの実体を取得.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                stacklessBvhNode.meshid = (float)item->geomid();

                if (isPrimitiveLeaf) {
                    // Leaves of this tree are primitive.
                    stacklessBvhNode.primid = (float)ctxt.findTriIdxFromPointer(item);
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

                stacklessBvhNode.boxmax_0 = left ? aten::vec4(left->getBoundingbox().maxPos(), 0) : aten::vec4(0);
                stacklessBvhNode.boxmin_0 = left ? aten::vec4(left->getBoundingbox().minPos(), 0) : aten::vec4(0);

                stacklessBvhNode.boxmax_1 = right ? aten::vec4(right->getBoundingbox().maxPos(), 0) : aten::vec4(0);
                stacklessBvhNode.boxmin_1 = right ? aten::vec4(right->getBoundingbox().minPos(), 0) : aten::vec4(0);

                stacklessBvhNode.child_0 = (float)(left ? left->getTraversalOrder() : -1);
                stacklessBvhNode.child_1 = (float)(right ? right->getTraversalOrder() : -1);
            }

            listStacklessBvhNode.push_back(stacklessBvhNode);
        }
    }

    bool StacklessBVH::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listStacklessBvhNode, r, t_min, t_max, isect);
    }

    bool StacklessBVH::hit(
        const context& ctxt,
        int exid,
        const std::vector<std::vector<StacklessBvhNode>>& listStacklessBvhNode,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        real hitt = AT_MATH_INF;

        int nodeid = 0;
        uint32_t bitstack = 0;

        for (;;) {
            const StacklessBvhNode* node = nullptr;

            if (nodeid >= 0) {
                node = &listStacklessBvhNode[exid][nodeid];
            }

            if (!node) {
                break;
            }

            bool isHit = false;

            if (node->isLeaf()) {
                Intersection isectTmp;

                auto s = ctxt.getTransformable((int)node->shapeid);

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

                    isHit = hit(
                        ctxt,
                        (int)node->exid,
                        listStacklessBvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else if (node->primid >= 0) {
                    // Hit test for a primitive.
                    auto prim = ctxt.getTriangle((int)node->primid);
                    isHit = prim->hit(ctxt, r, t_min, t_max, isectTmp);
                    if (isHit) {
                        isectTmp.objid = s->id();
                    }
                }
                else {
                    // Hit test for a shape.
                    isHit = s->hit(ctxt, r, t_min, t_max, isectTmp);
                }

                if (isHit) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;
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
                        nodeid = (int)(t[0] < t[1] ? node->child_0 : node->child_1);
                        bitstack = bitstack | 1;
                    }
                    else if (hit[0]) {
                        nodeid = (int)node->child_0;
                    }
                    else if (hit[1]) {
                        nodeid = (int)node->child_1;
                    }

                    continue;
                }
            }

            while ((bitstack & 1) == 0) {
                if (bitstack == 0) {
                    return (isect.objid >= 0);
                }

                nodeid = (int)node->parent;
                bitstack = bitstack >> 1;

                node = &listStacklessBvhNode[exid][nodeid];
            }

            nodeid = (int)node->sibling;
            bitstack = bitstack ^ 1;
        }

        return (isect.objid >= 0);
    }
}
