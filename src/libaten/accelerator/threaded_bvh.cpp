#include <random>
#include <vector>
#include <iterator>

#include "accelerator/threaded_bvh.h"
#include "accelerator/bvh.h"
#include "accelerator/bvh_util.h"
#include "geometry/transformable.h"
#include "geometry/PolygonObject.h"

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
        if (is_nested_) {
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
        AT_ASSERT(is_nested_);

        // Build as the simple BVH.
        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());

        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

        // Register bvh nodes tree to linear list temporarily by traversing bvhnode.
        registerBvhNodeToLinearList(
            ctxt,
            m_bvh.getRoot(),
            threadedBvhNodeEntries);

        std::vector<int32_t> listParentId;
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

    // Build all objects in the top layer tree.
    void ThreadedBVH::buildAsTopLayerTree(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        AT_ASSERT(!is_nested_);

        // Build as the simple BVH.
        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());

        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

        // Register bvh nodes tree to linear list temporarily by traversing bvhnode.
        registerBvhNodeToLinearList(
            ctxt,
            m_bvh.getRoot(),
            threadedBvhNodeEntries);

        // Bottom layer BVHs are stored as map.
        // In registerBvhNodeToLinearList, we can't register it to the linear list in order.
        // So, we registered it temporarily in map and we expand it and register it to the linear list.
        if (!m_mapNestedBvh.empty()) {
            // Allocate the linear list.
            m_nestedBvh.resize(m_mapNestedBvh.size());

            for (auto it = m_mapNestedBvh.begin(); it != m_mapNestedBvh.end(); it++) {
                int32_t exid = it->first;
                auto accel = it->second;

                // NOTE:
                // 0 is used for the top layer.
                m_nestedBvh[exid - 1] = accel;
            }

            m_mapNestedBvh.clear();
        }

        std::vector<int32_t> listParentId;

        if (m_enableLayer) {
            // NOTE
            // 0 is for top layer. So, need +1.
            m_listThreadedBvhNode.resize(m_nestedBvh.size() + 1);
        }
        else {
            m_listThreadedBvhNode.resize(1);
        }

        // Register bvh nodes from the temporal list to the actual list with filling the information.
        registerThreadedBvhNode(
            ctxt,
            false,
            threadedBvhNodeEntries,
            m_listThreadedBvhNode[0],
            listParentId);

        // Set traverse order to linear bvh.
        setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);

        // To make one linear list to include all nodes regardless of top or bottom,
        // combine nested (bottom layer) threaded bvh nodes to top layer linear list.
        if (m_enableLayer) {
            for (int32_t i = 0; i < m_nestedBvh.size(); i++) {
                auto node = m_nestedBvh[i];

                // TODO
                if (node->getAccelType() == AccelType::ThreadedBvh) {
                    auto threadedBvh = static_cast<ThreadedBVH*>(node);

                    auto& linear_list_threaded_bvh_nodes = threadedBvh->m_listThreadedBvhNode[0];

                    // NODE
                    // linear_list_threaded_bvh_nodes[0] is for top layer.
                    std::copy(
                        linear_list_threaded_bvh_nodes.begin(),
                        linear_list_threaded_bvh_nodes.end(),
                        std::back_inserter(m_listThreadedBvhNode[i + 1]));
                }
            }
        }

        //dump(m_listThreadedBvhNode[1], "node.txt");
    }

    void ThreadedBVH::dump(std::vector<ThreadedBvhNode>& nodes, std::string_view path)
    {
        FILE* fp = fopen(path.data(), "wt");
        if (!fp) {
            AT_ASSERT(false);
            return;
        }

        for (const auto& n : nodes) {
            fprintf(fp, "%d %d %d %d %d %d (%.3f, %.3f, %.3f) (%.3f, %.3f, %.3f)\n",
                (int32_t)n.hit, (int32_t)n.miss, (int32_t)n.object_id, (int32_t)n.primid, (int32_t)n.exid, (int32_t)n.meshid,
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
        std::vector<int32_t>& listParentId)
    {
        // The nods in threadedBvhNodeEntries are still the temporal node.
        // We need to fill the necessary information for the actual usage.
        // Therefore, we retrieve the node from threadedBvhNodeEntries
        // and fill the information to the node and store it to threadedBvhNodes.

        // Allocate lists.
        threadedBvhNodes.reserve(threadedBvhNodeEntries.size());
        listParentId.reserve(threadedBvhNodeEntries.size());

        for (const auto& entry : threadedBvhNodeEntries) {
            auto node = entry.node;

            ThreadedBvhNode real_node;

            auto bbox = node->getBoundingbox();
            bbox = aten::aabb::transform(bbox, entry.mtx_L2W);

            // Keep idx of node's parent.
            // The node is stored linearly in order.
            // Therefore, the order to store the idx of node's parent is the same as the order in the node linear list.
            auto parent = node->getParent();
            int32_t parentId = parent ? parent->getTraversalOrder() : -1;
            listParentId.push_back(parentId);

            if (node->isLeaf()) {
                hitable* item = node->getItem();

                // Obtain self index.
                real_node.object_id = static_cast<float>(ctxt.FindTransformableIdxFromPointer(item));

                // Get the actual object entity.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                real_node.meshid = static_cast<float>(item->GetMeshId());

                if (isPrimitiveLeaf) {
                    // Leaves of this tree are primitive.
                    real_node.primid = static_cast<float>(ctxt.FindTriangleIdxFromPointer(item));
                    real_node.exid = -1.0f;
                }
                else {
                    // This node specifies to another bvh tree.
                    auto exid = node->getExternalId();

                    // subexid means LOD.
                    auto subexid = node->getSubExternalId();

                    if (exid >= 0) {
                        real_node.noExternal = false;
                        real_node.hasLod = (subexid >= 0);
                        real_node.mainExid = exid;
                        real_node.lodExid = (subexid >= 0 ? subexid : 0);
                    }
                    else {
                        // In this case, the item which node keeps is sphere/cube.
                        real_node.exid = -1.0f;
                    }
                }
            }

            real_node.boxmax = aten::vec4(bbox.maxPos(), 0);
            real_node.boxmin = aten::vec4(bbox.minPos(), 0);

            // Register to the actual bvh linear list.
            threadedBvhNodes.push_back(real_node);
        }
    }

    void ThreadedBVH::setOrder(
        const std::vector<ThreadedBvhNodeEntry>& threadedBvhNodeEntries,
        const std::vector<int32_t>& listParentId,
        std::vector<ThreadedBvhNode>& threadedBvhNodes)
    {
        // Threaded BVH has the hit/miss instead of left/right.
        // And, for GPU, how to specify such kind of node should not be the poniter but the index.
        // In this API, we fill the index.

        auto num = threadedBvhNodes.size();

        for (int32_t n = 0; n < num; n++) {
            auto entry_node = threadedBvhNodeEntries[n].node;
            auto& real_node = threadedBvhNodes[n];

            bvhnode* next = nullptr;
            if (n + 1 < num) {
                next = threadedBvhNodeEntries[n + 1].node;
            }

            if (entry_node->isLeaf()) {
                // Hit/Miss.
                // Always the next node in the array.
                if (next) {
                    real_node.hit = static_cast<float>(next->getTraversalOrder());
                    real_node.miss = static_cast<float>(next->getTraversalOrder());
                }
                else {
                    real_node.hit = -1;
                    real_node.miss = -1;
                }
            }
            else {
                // Hit.
                // Always the next node in the array.
                if (next) {
                    real_node.hit = static_cast<float>(next->getTraversalOrder());
                }
                else {
                    real_node.hit = -1;
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

                    bool isLeft = (left == entry_node);

                    if (isLeft) {
                        // Internal, left: sibling node.
                        auto sibling = right;
                        isLeft = (sibling != nullptr);

                        if (isLeft) {
                            real_node.miss = static_cast<float>(sibling->getTraversalOrder());
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
                                        real_node.miss = (float)sibling->getTraversalOrder();
                                        break;
                                    }
                                }
                            }
                            else {
                                real_node.miss = -1;
                                break;
                            }

                            curParent = grandParent;
                        }
                    }
                }
                else {
                    real_node.miss = -1;
                }
            }
        }
    }

    bool ThreadedBVH::hit(
        const context& ctxt,
        const ray& r,
        float t_min, float t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listThreadedBvhNode, r, t_min, t_max, isect);
    }

    bool ThreadedBVH::hit(
        const context& ctxt,
        int32_t exid,
        const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
        const ray& r,
        float t_min, float t_max,
        Intersection& isect,
        aten::HitStopType hit_stop_type/*= aten::HitStopType::Closest*/) const
    {
        float hitt = AT_MATH_INF;

        int32_t nodeid = 0;

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

                auto s = node->object_id >= 0 ? ctxt.GetTransformable((int32_t)node->object_id) : nullptr;

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

                    //int32_t exid = node->mainExid;
                    int32_t exid = *(int32_t*)(&node->exid);
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
                    auto prim = ctxt.GetTriangleInstance((int32_t)node->primid);
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
                isHit = aten::aabb::hit(r, node->boxmin, node->boxmax, t_min, t_max);
            }

            if (isHit) {
                nodeid = (int32_t)node->hit;
            }
            else {
                nodeid = (int32_t)node->miss;
            }
        }

        return (isect.objid >= 0);
    }

    void ThreadedBVH::update(const context& ctxt)
    {
        m_bvh.update(ctxt);

        setBoundingBox(m_bvh.getBoundingbox());

        auto root = m_bvh.getRoot();
        std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;
        registerBvhNodeToLinearList(ctxt, root, threadedBvhNodeEntries);

        std::vector<int32_t> listParentId;
        listParentId.reserve(threadedBvhNodeEntries.size());

        m_listThreadedBvhNode[0].clear();

        for (auto& entry : threadedBvhNodeEntries) {
            auto node = entry.node;

            ThreadedBvhNode real_node;

            // NOTE
            // Differ set hit/miss index.

            auto bbox = node->getBoundingbox();

            auto parent = node->getParent();
            int32_t parentId = parent ? parent->getTraversalOrder() : -1;
            listParentId.push_back(parentId);

            if (node->isLeaf()) {
                hitable* item = node->getItem();

                // 自分自身のIDを取得.
                real_node.object_id = (float)ctxt.FindTransformableIdxFromPointer(item);
                AT_ASSERT(real_node.object_id >= 0);

                // インスタンスの実体を取得.
                auto internalObj = item->getHasObject();

                if (internalObj) {
                    item = const_cast<hitable*>(internalObj);
                }

                real_node.meshid = (float)item->GetMeshId();

                int32_t exid = node->getExternalId();
                int32_t subexid = node->getSubExternalId();

                if (exid < 0) {
                    real_node.exid = -1.0f;
                }
                else {
                    real_node.noExternal = false;
                    real_node.hasLod = (subexid >= 0);
                    real_node.mainExid = (exid >= 0 ? exid : 0);
                    real_node.lodExid = (subexid >= 0 ? subexid : 0);
                }
            }

            real_node.boxmax = aten::vec4(bbox.maxPos(), 0);
            real_node.boxmin = aten::vec4(bbox.minPos(), 0);

            m_listThreadedBvhNode[0].push_back(real_node);
        }

        setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);
    }

    // Register bvh nodes tree to linear list by traversing bvh.
    void ThreadedBVH::registerBvhNodeToLinearList(
        const context& ctxt,
        bvhnode* node,
        std::vector<ThreadedBvhNodeEntry>& nodes)
    {
        if (!node) {
            return;
        }

        // Get traversal order.
        // The nodes are registered linearly.
        // So, just the size of the linear node list is the order.
        int32_t order = static_cast<int32_t>(nodes.size());
        node->setTraversalOrder(order);

        mat4 mtx_L2W;

        // There is a possibility that the node doesn't have the actual entity.
        // e.g. root of the tree.
        auto item = node->getItem();
        auto idx = ctxt.FindTransformableIdxFromPointer(item);

        if (idx >= 0) {
            auto t = ctxt.GetTransformable(idx);

            mat4 mtx_W2L;
            t->getMatrices(mtx_L2W, mtx_W2L);

            if (t->GetParam().type == ObjectType::Instance) {
                // TODO
                auto obj = const_cast<hitable*>(t->getHasObject());

                // In this case, subobj means LOD of the actual obj.
                auto subobj = const_cast<hitable*>(t->getHasSecondObject());

                // exid is idx for bottom later BVH.
                // In this timing, we can't get its index in order.
                // NOTE:
                // 0 is for top layer, so need to add 1.
                int32_t exid = ctxt.FindPolygonalTransformableOrderFromPointer(obj) + 1;
                int32_t subexid = subobj ? ctxt.FindPolygonalTransformableOrderFromPointer(subobj) + 1 : -1;

                node->setExternalId(exid);
                node->setSubExternalId(subexid);

                // 
                auto accel = obj->getInternalAccelerator();

                // Keep nested (bottom layer) BVH in map.
                // In this timing, we can't get its index in order.
                // It means we can't register it in the linear list directly.
                // So, we need to register it to map temporariy as the pair of the idx and the bottom layer.
                // It will be expanded and registered to the linear list later.
                if (m_mapNestedBvh.find(exid) == m_mapNestedBvh.end()) {
                    m_mapNestedBvh.insert(std::pair<int32_t, accelerator*>(exid, accel));
                }

                // If there is LOD.
                if (subobj) {
                    accel = subobj->getInternalAccelerator();

                    if (m_mapNestedBvh.find(subexid) == m_mapNestedBvh.end()) {
                        m_mapNestedBvh.insert(std::pair<int32_t, accelerator*>(subexid, accel));
                    }
                }
            }
        }

        // Register linearly.
        // But, in this case, the registered element is still temporal one.
        // It will be converted to the finalized one later.
        nodes.push_back(ThreadedBvhNodeEntry{ node, mtx_L2W });

        // Traverse to left and right recursively.
        registerBvhNodeToLinearList(ctxt, node->getLeft(), nodes);
        registerBvhNodeToLinearList(ctxt, node->getRight(), nodes);
    }
}
