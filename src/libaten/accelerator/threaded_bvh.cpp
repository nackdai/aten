#include <random>
#include <vector>
#include <iterator>

#include "accelerator/threaded_bvh.h"
#include "accelerator/bvh.h"
#include "geometry/transformable.h"
#include "geometry/object.h"

//#pragma optimize( "", off)

// Threaded BVH
// http://www.ci.i.u-tokyo.ac.jp/~hachisuka/tdf2015.pdf

namespace aten
{
    void ThreadedBVH::build(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        if (m_isNested) {
            buildAsNestedTree(ctxt, list, num, bbox);
        }
        else {
            buildAsTopLayerTree(ctxt, list, num, bbox);
        }
    }

    void ThreadedBVH::buildAsNestedTree(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        AT_ASSERT(m_isNested);

        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());

        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

        // Convert to linear list.
        registerBvhNodeToLinearList(
            ctxt,
            m_bvh.getRoot(),
            threadedBvhNodeEntries);

        std::vector<int> listParentId;
        m_listThreadedBvhNode.resize(1);

        // Register bvh node for gpu.
        registerThreadedBvhNode(
            ctxt,
            true,
            threadedBvhNodeEntries,
            m_listThreadedBvhNode[0],
            listParentId);

        // Set order.
        setOrder(
            threadedBvhNodeEntries,
            listParentId,
            m_listThreadedBvhNode[0]);
    }

    void ThreadedBVH::buildAsTopLayerTree(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        AT_ASSERT(!m_isNested);

        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());

        // Gather local-world matrix.
        ctxt.copyMatricesAndUpdateTransformableMatrixIdx(m_mtxs);

        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

        // Register to linear list to traverse bvhnode easily.
        registerBvhNodeToLinearList(
            ctxt,
            m_bvh.getRoot(),
            threadedBvhNodeEntries);

        // Convert from map to vector.
        if (!m_mapNestedBvh.empty()) {
            m_nestedBvh.resize(m_mapNestedBvh.size());

            for (auto it = m_mapNestedBvh.begin(); it != m_mapNestedBvh.end(); it++) {
                int exid = it->first;
                auto accel = it->second;

                // NOTE
                // 0 は上位レイヤーで使用しているので、-1する.
                m_nestedBvh[exid - 1] = accel;
            }

            m_mapNestedBvh.clear();
        }

        std::vector<int> listParentId;

        if (m_enableLayer) {
            // NOTE
            // 0 is for top layer. So, need +1.
            m_listThreadedBvhNode.resize(m_nestedBvh.size() + 1);
        }
        else {
            m_listThreadedBvhNode.resize(1);
        }

        // Register bvh node for gpu.
        registerThreadedBvhNode(
            ctxt,
            false,
            threadedBvhNodeEntries,
            m_listThreadedBvhNode[0],
            listParentId);

        // Set traverse order for linear bvh.
        setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);

        // Copy nested threaded bvh nodes to top layer tree.
        if (m_enableLayer) {
            for (int i = 0; i < m_nestedBvh.size(); i++) {
                auto node = m_nestedBvh[i];

                // TODO
                if (node->getAccelType() == AccelType::ThreadedBvh) {
                    auto threadedBvh = static_cast<ThreadedBVH*>(node);

                    auto& threadedBvhNodes = threadedBvh->m_listThreadedBvhNode[0];

                    // NODE
                    // m_listThreadedBvhNode[0] is for top layer.
                    std::copy(
                        threadedBvhNodes.begin(), 
                        threadedBvhNodes.end(), 
                        std::back_inserter(m_listThreadedBvhNode[i + 1]));
                }
            }
        }

        //dump(m_listThreadedBvhNode[1], "node.txt");
    }

    void ThreadedBVH::dump(std::vector<ThreadedBvhNode>& nodes, const char* path)
    {
        FILE* fp = fopen(path, "wt");
        if (!fp) {
            AT_ASSERT(false);
            return;
        }

        for (const auto& n : nodes) {
            fprintf(fp, "%d %d %d %d %d %d (%.3f, %.3f, %.3f) (%.3f, %.3f, %.3f)\n", 
                (int)n.hit, (int)n.miss, (int)n.shapeid, (int)n.primid, (int)n.exid, (int)n.meshid,
                n.boxmin.x, n.boxmin.y, n.boxmin.z,
                n.boxmax.x, n.boxmax.y, n.boxmax.z);
        }

        fclose(fp);
    }

    void ThreadedBVH::registerThreadedBvhNode(
        const context& ctxt,
        bool isPrimitiveLeaf,
        const std::vector<ThreadedBvhNodeEntry>& threadedBvhNodeEntries,
        std::vector<ThreadedBvhNode>& threadedBvhNodes,
        std::vector<int>& listParentId)
    {
        threadedBvhNodes.reserve(threadedBvhNodeEntries.size());
        listParentId.reserve(threadedBvhNodeEntries.size());

        for (const auto& entry : threadedBvhNodeEntries) {
            auto node = entry.node;

            ThreadedBvhNode gpunode;

            // NOTE
            // Differ set hit/miss index.

            auto bbox = node->getBoundingbox();
            bbox = aten::aabb::transform(bbox, entry.mtxL2W);

            auto parent = node->getParent();
            int parentId = parent ? parent->getTraversalOrder() : -1;
            listParentId.push_back(parentId);

            if (node->isLeaf()) {
                hitable* item = node->getItem();

                // 自分自身のIDを取得.
                gpunode.shapeid = (float)ctxt.findTransformableIdxFromPointer(item);

                // インスタンスの実体を取得.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                gpunode.meshid = (float)item->geomid();

                if (isPrimitiveLeaf) {
                    // Leaves of this tree are primitive.
                    gpunode.primid = (float)ctxt.findTriIdxFromPointer(item);
                    gpunode.exid = -1.0f;
                }
                else {
                    auto exid = node->getExternalId();
                    auto subexid = node->getSubExternalId();

                    if (exid >= 0) {
                        gpunode.noExternal = false;
                        gpunode.hasLod = (subexid >= 0);
                        gpunode.mainExid = exid;
                        gpunode.lodExid = (subexid >= 0 ? subexid : 0);
                    }
                    else {
                        // In this case, the item which node keeps is sphere/cube.
                        gpunode.exid = -1.0f;
                    }
                }
            }

            gpunode.boxmax = aten::vec4(bbox.maxPos(), 0);
            gpunode.boxmin = aten::vec4(bbox.minPos(), 0);

            threadedBvhNodes.push_back(gpunode);
        }
    }

    void ThreadedBVH::setOrder(
        const std::vector<ThreadedBvhNodeEntry>& threadedBvhNodeEntries,
        const std::vector<int>& listParentId,
        std::vector<ThreadedBvhNode>& threadedBvhNodes)
    {
        auto num = threadedBvhNodes.size();

        for (int n = 0; n < num; n++) {
            auto node = threadedBvhNodeEntries[n].node;
            auto& gpunode = threadedBvhNodes[n];

            bvhnode* next = nullptr;
            if (n + 1 < num) {
                next = threadedBvhNodeEntries[n + 1].node;
            }

            if (node->isLeaf()) {
                // Hit/Miss.
                // Always the next node in the array.
                if (next) {
                    gpunode.hit = (float)next->getTraversalOrder();
                    gpunode.miss = (float)next->getTraversalOrder();
                }
                else {
                    gpunode.hit = -1;
                    gpunode.miss = -1;
                }
            }
            else {
                // Hit.
                // Always the next node in the array.
                if (next) {
                    gpunode.hit = (float)next->getTraversalOrder();
                }
                else {
                    gpunode.hit = -1;
                }

                // Miss.

                // Search the parent.
                auto parentId = listParentId[n];
                bvhnode* parent = (parentId >= 0
                    ? threadedBvhNodeEntries[parentId].node
                    : nullptr);

                if (parent) {
                    bvhnode* left = parent->getLeft();
                    bvhnode* right = parent->getRight();

                    bool isLeft = (left == node);

                    if (isLeft) {
                        // Internal, left: sibling node.
                        auto sibling = right;
                        isLeft = (sibling != nullptr);

                        if (isLeft) {
                            gpunode.miss = (float)sibling->getTraversalOrder();
                        }
                    }

                    bvhnode* curParent = parent;

                    if (!isLeft) {
                        // Internal, right: parent's sibling node (until it exists) .
                        for (;;) {
                            // Search the grand parent.
                            auto grandParentId = listParentId[curParent->getTraversalOrder()];
                            bvhnode* grandParent = (grandParentId >= 0
                                ? threadedBvhNodeEntries[grandParentId].node
                                : nullptr);

                            if (grandParent) {
                                bvhnode* _left = grandParent->getLeft();
                                bvhnode* _right = grandParent->getRight();

                                auto sibling = _right;
                                if (sibling) {
                                    if (sibling != curParent) {
                                        gpunode.miss = (float)sibling->getTraversalOrder();
                                        break;
                                    }
                                }
                            }
                            else {
                                gpunode.miss = -1;
                                break;
                            }

                            curParent = grandParent;
                        }
                    }
                }
                else {
                    gpunode.miss = -1;
                }
            }
        }
    }

    bool ThreadedBVH::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listThreadedBvhNode, r, t_min, t_max, isect);
    }

    bool ThreadedBVH::hit(
        const context& ctxt,
        int exid,
        const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        real hitt = AT_MATH_INF;

        int nodeid = 0;

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

                auto s = node->shapeid >= 0 ? ctxt.getTransformable((int)node->shapeid) : nullptr;

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

                    //int exid = node->mainExid;
                    int exid = *(int*)(&node->exid);
                    exid = AT_BVHNODE_MAIN_EXID(exid);

                    isHit = hit(
                        ctxt,
                        exid,
                        listThreadedBvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);

                    if (isHit) {
                        isectTmp.objid = s->id();
                    }
                }
                else if (node->primid >= 0) {
                    // Hit test for a primitive.
                    auto prim = ctxt.getTriangle((int)node->primid);
                    isHit = prim->hit(ctxt, r, t_min, t_max, isectTmp);
                    if (isHit) {
                        // Set dummy to return if ray hit.
                        isectTmp.objid = s ? s->id() : 1;
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
                isHit = aten::aabb::hit(r, node->boxmin, node->boxmax, t_min, t_max);
            }

            if (isHit) {
                nodeid = (int)node->hit;
            }
            else {
                nodeid = (int)node->miss;
            }
        }

        return (isect.objid >= 0);
    }

    void ThreadedBVH::update(const context& ctxt)
    {
        m_bvh.update();

        setBoundingBox(m_bvh.getBoundingbox());

        // TODO
        // More efficient. ex) Gather only transformed object etc...
        // Gather local-world matrix.
        m_mtxs.clear();
        ctxt.copyMatricesAndUpdateTransformableMatrixIdx(m_mtxs);

        auto root = m_bvh.getRoot();
        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;
        registerBvhNodeToLinearList(ctxt, root, threadedBvhNodeEntries);

        std::vector<int> listParentId;
        listParentId.reserve(threadedBvhNodeEntries.size());

        m_listThreadedBvhNode[0].clear();

        for (auto& entry : threadedBvhNodeEntries) {
            auto node = entry.node;

            ThreadedBvhNode gpunode;

            // NOTE
            // Differ set hit/miss index.

            auto bbox = node->getBoundingbox();

            auto parent = node->getParent();
            int parentId = parent ? parent->getTraversalOrder() : -1;
            listParentId.push_back(parentId);

            if (node->isLeaf()) {
                hitable* item = node->getItem();

                // 自分自身のIDを取得.
                gpunode.shapeid = (float)ctxt.findTransformableIdxFromPointer(item);
                AT_ASSERT(gpunode.shapeid >= 0);

                // インスタンスの実体を取得.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                gpunode.meshid = (float)item->geomid();

                int exid = node->getExternalId();
                int subexid = node->getSubExternalId();

                if (exid < 0) {
                    gpunode.exid = -1.0f;
                }
                else {
                    gpunode.noExternal = false;
                    gpunode.hasLod = (subexid >= 0);
                    gpunode.mainExid = (exid >= 0 ? exid : 0);
                    gpunode.lodExid = (subexid >= 0 ? subexid : 0);
                }
            }

            gpunode.boxmax = aten::vec4(bbox.maxPos(), 0);
            gpunode.boxmin = aten::vec4(bbox.minPos(), 0);

            m_listThreadedBvhNode[0].push_back(gpunode);
        }

        setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);
    }

    void ThreadedBVH::registerBvhNodeToLinearList(
        const context& ctxt,
        bvhnode* node,
        std::vector<ThreadedBvhNodeEntry>& nodes)
    {
        if (!node) {
            return;
        }

        int order = nodes.size();
        node->setTraversalOrder(order);

        mat4 mtxL2W;

        auto item = node->getItem();
        auto idx = ctxt.findTransformableIdxFromPointer(item);

        if (idx >= 0) {
            auto t = ctxt.getTransformable(idx);

            mat4 mtxW2L;
            t->getMatrices(mtxL2W, mtxW2L);

            if (t->getParam().type == GeometryType::Instance) {
                // TODO
                auto obj = const_cast<hitable*>(t->getHasObject());
                auto subobj = const_cast<hitable*>(t->getHasSecondObject());

                // NOTE
                // 0 is for top layer, so need to add 1.
                int exid = ctxt.findPolygonalTransformableOrderFromPointer(obj) + 1;
                int subexid = subobj ? ctxt.findPolygonalTransformableOrderFromPointer(subobj) + 1 : -1;

                node->setExternalId(exid);
                node->setSubExternalId(subexid);

                auto accel = obj->getInternalAccelerator();

                // Keep nested bvh.
                if (m_mapNestedBvh.find(exid) == m_mapNestedBvh.end()) {
                    m_mapNestedBvh.insert(std::pair<int, accelerator*>(exid, accel));
                }
                
                if (subobj) {
                    accel = subobj->getInternalAccelerator();

                    if (m_mapNestedBvh.find(subexid) == m_mapNestedBvh.end()) {
                        m_mapNestedBvh.insert(std::pair<int, accelerator*>(subexid, accel));
                    }
                }
            }
        }

        nodes.push_back(ThreadedBvhNodeEntry(node, mtxL2W));

        registerBvhNodeToLinearList(ctxt, node->getLeft(), nodes);
        registerBvhNodeToLinearList(ctxt, node->getRight(), nodes);
    }
}
