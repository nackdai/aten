#include "accelerator/stackless_qbvh.h"
#include "accelerator/bvh_util.h"

//#pragma optimize( "", off)

// NOTE
// Stackless Multi-BVH Traversal for CPU, MIC and GPU Ray Tracing
// http://cg.iit.bme.hu/~afra/publications/afra2013cgf_mbvhsl.pdf

namespace aten
{
    void StacklessQbvh::build(
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

        std::vector<std::vector<BvhNode>> listBvhNode;

        // Register to linear list to traverse bvhnode easily.
        auto root = m_bvh.getRoot();
        listBvhNode.push_back(std::vector<BvhNode>());
        registerBvhNodeToLinearList(root, nullptr, nullptr, aten::mat4::Identity, listBvhNode[0], listBvh, nestedBvhMap);

        for (int32_t i = 0; i < listBvh.size(); i++) {
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

            listBvhNode.push_back(std::vector<BvhNode>());
            std::vector<accelerator*> dummy;

            registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1], dummy, nestedBvhMap);
            AT_ASSERT(dummy.empty());
        }

        m_listQbvhNode.resize(listBvhNode.size());

        // Convert to QBVH.
        for (int32_t i = 0; i < listBvhNode.size(); i++) {
            bool isPrimitiveLeafBvh = (i > 0);

            auto numNodes = convertFromBvh(
                ctxt,
                isPrimitiveLeafBvh,
                listBvhNode[i],
                m_listQbvhNode[i]);
        }
    }

    void StacklessQbvh::registerBvhNodeToLinearList(
        bvhnode* root,
        bvhnode* parentNode,
        hitable* nestParent,
        const aten::mat4& mtxL2W,
        std::vector<BvhNode>& listBvhNode,
        std::vector<accelerator*>& listBvh,
        std::map<hitable*, accelerator*>& nestedBvhMap)
    {
        registerBvhNodeToLinearListRecursively<BvhNode>(
            root,
            parentNode,
            nestParent,
            mtxL2W,
            listBvhNode,
            listBvh,
            nestedBvhMap,
            [this](std::vector<BvhNode>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
        {
            list.push_back(BvhNode(node, obj, mtx));
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

    uint32_t StacklessQbvh::convertFromBvh(
        const context& ctxt,
        bool isPrimitiveLeaf,
        std::vector<BvhNode>& listBvhNode,
        std::vector<StacklessQbvhNode>& listQbvhNode)
    {
        struct QbvhStackEntry {
            uint32_t qbvhNodeIdx{ 0 };
            uint32_t bvhNodeIdx{ 0 };
            int32_t parentQbvhNodeIdx{ -1 };

            QbvhStackEntry(uint32_t StacklessQbvh, uint32_t bvh, int32_t parent)
                : qbvhNodeIdx(StacklessQbvh), bvhNodeIdx(bvh), parentQbvhNodeIdx(parent)
            {}
            QbvhStackEntry() {}
        };

        listQbvhNode.reserve(listBvhNode.size());
        listQbvhNode.push_back(StacklessQbvhNode());

        QbvhStackEntry stack[256];
        stack[0] = QbvhStackEntry(0, 0, -1);

        int32_t stackPos = 1;
        uint32_t numNodes = 1;

        int32_t children[4];

        while (stackPos > 0) {
            auto top = stack[--stackPos];

            auto& qbvhNode = listQbvhNode[top.qbvhNodeIdx];
            const auto& bvhNode = listBvhNode[top.bvhNodeIdx];

            qbvhNode.parent = (float)top.parentQbvhNodeIdx;

            if (top.qbvhNodeIdx == 52005) {
                int32_t xxxx = 0;
            }

            if (top.parentQbvhNodeIdx >= 0) {
                const auto& parentQbvhNode = listQbvhNode[top.parentQbvhNodeIdx];
                int32_t leftSiblingIdx = (int32_t)parentQbvhNode.leftChildrenIdx;
                int32_t siblingNum = (int32_t)parentQbvhNode.numChildren;

                // 基準点を決める.
                int32_t base = 0;
                for (int32_t i = 0; i < 4; i++) {
                    if (top.qbvhNodeIdx == leftSiblingIdx + i) {
                        base = i + 1;
                        break;
                    }
                }

                if (siblingNum == 4) {
                    qbvhNode.sib[0] = (float)(leftSiblingIdx + ((base + 0) % siblingNum));
                    qbvhNode.sib[1] = (float)(leftSiblingIdx + ((base + 1) % siblingNum));
                    qbvhNode.sib[2] = (float)(leftSiblingIdx + ((base + 2) % siblingNum));
                }
                else if (siblingNum == 3) {
                    qbvhNode.sib[0] = (float)(leftSiblingIdx + ((base + 0) % siblingNum));
                    qbvhNode.sib[1] = (float)(leftSiblingIdx + ((base + 1) % siblingNum));
                    qbvhNode.sib[2] = (float)(leftSiblingIdx + ((base + 2) % siblingNum));

                    switch (base) {
                    case 1:
                        qbvhNode.sib[0] = (float)(leftSiblingIdx + ((base + 0) % siblingNum));
                        qbvhNode.sib[1] = (float)(leftSiblingIdx + ((base + 1) % siblingNum));
                        qbvhNode.sib[2] = -1.0f;
                        break;
                    case 2:
                        qbvhNode.sib[0] = (float)(leftSiblingIdx + ((base + 0) % siblingNum));
                        qbvhNode.sib[1] = -1.0f;
                        qbvhNode.sib[2] = (float)(leftSiblingIdx + ((base + 1) % siblingNum));
                        break;
                    case 3:
                        qbvhNode.sib[0] = -1.0f;
                        qbvhNode.sib[1] = (float)(leftSiblingIdx + ((base + 0) % siblingNum));
                        qbvhNode.sib[2] = (float)(leftSiblingIdx + ((base + 1) % siblingNum));
                        break;
                    }
                }
                else if (siblingNum == 2) {
                    qbvhNode.sib[0] = -1.0f;
                    qbvhNode.sib[1] = -1.0f;
                    qbvhNode.sib[2] = -1.0f;

                    switch (base) {
                    case 1:
                        qbvhNode.sib[0] = (float)(leftSiblingIdx + (base % siblingNum));
                        break;
                    case 2:
                        qbvhNode.sib[2] = (float)(leftSiblingIdx + (base % siblingNum));
                        break;
                    }
                }
                else {
                    qbvhNode.sib[0] = -1.0f;
                    qbvhNode.sib[1] = -1.0f;
                    qbvhNode.sib[2] = -1.0f;
                }
            }

            int32_t numChildren = getChildren(listBvhNode, top.bvhNodeIdx, children);

            if (numChildren == 0) {
                // No children, so it is a leaf.
                setQbvhNodeLeafParams(ctxt, isPrimitiveLeaf, bvhNode, qbvhNode);
                continue;
            }

            // Fill node.
            fillQbvhNode(
                qbvhNode,
                listBvhNode,
                children,
                numChildren);

            qbvhNode.leftChildrenIdx = (float)numNodes;
            qbvhNode.numChildren = (float)numChildren;

            // push all children to the stack
            for (int32_t i = 0; i < numChildren; i++) {
                stack[stackPos++] = QbvhStackEntry(numNodes, children[i], top.qbvhNodeIdx);

                listQbvhNode.push_back(StacklessQbvhNode());
                ++numNodes;
            }
        }

        return numNodes;
    }

    void StacklessQbvh::setQbvhNodeLeafParams(
        const context& ctxt,
        bool isPrimitiveLeaf,
        const BvhNode& bvhNode,
        StacklessQbvhNode& qbvhNode)
    {
        auto node = bvhNode.node;
        auto nestParent = bvhNode.nestParent;

        auto bbox = node->getBoundingbox();
        bbox = aten::aabb::transform(bbox, bvhNode.mtxL2W);

        qbvhNode.leftChildrenIdx = 0;

        qbvhNode.bmaxx.set(real(0));
        qbvhNode.bmaxy.set(real(0));
        qbvhNode.bmaxz.set(real(0));

        qbvhNode.bminx.set(real(0));
        qbvhNode.bminy.set(real(0));
        qbvhNode.bminz.set(real(0));

        qbvhNode.numChildren = 0;

        if (node->isLeaf()) {
            hitable* item = node->getItem();

            // 自分自身のIDを取得.
            qbvhNode.object_id = (float)ctxt.findTransformableIdxFromPointer(item);

            // もしなかったら、ネストしているので親のIDを取得.
            if (qbvhNode.object_id < 0) {
                if (nestParent) {
                    qbvhNode.object_id = (float)ctxt.findTransformableIdxFromPointer(nestParent);
                }
            }

            // インスタンスの実体を取得.
            auto internalObj = item->getHasObject();

            if (internalObj) {
                item = const_cast<hitable*>(internalObj);
            }

            qbvhNode.meshid = (float)item->mesh_id();

            if (isPrimitiveLeaf) {
                // Leaves of this tree are primitive.
                qbvhNode.primid = (float)ctxt.findTriIdxFromPointer(item);
                qbvhNode.exid = -1.0f;
            }
            else {
                qbvhNode.exid = (float)node->getExternalId();
            }

            qbvhNode.isLeaf = (float)true;
        }
    }

    void StacklessQbvh::fillQbvhNode(
        StacklessQbvhNode& qbvhNode,
        std::vector<BvhNode>& listBvhNode,
        int32_t children[4],
        int32_t numChildren)
    {
        for (int32_t i = 0; i < numChildren; i++) {
            int32_t childIdx = children[i];
            const auto& bvhNode = listBvhNode[childIdx];

            const auto node = bvhNode.node;

            auto bbox = node->getBoundingbox();
            bbox = aten::aabb::transform(bbox, bvhNode.mtxL2W);

            const auto& bmax = bbox.maxPos();
            const auto& bmin = bbox.minPos();

            qbvhNode.bmaxx[i] = bmax.x;
            qbvhNode.bmaxy[i] = bmax.y;
            qbvhNode.bmaxz[i] = bmax.z;

            qbvhNode.bminx[i] = bmin.x;
            qbvhNode.bminy[i] = bmin.y;
            qbvhNode.bminz[i] = bmin.z;
        }

        // Set 0s for empty child.
        for (int32_t i = numChildren; i < 4; i++) {
            qbvhNode.bmaxx[i] = real(0);
            qbvhNode.bmaxy[i] = real(0);
            qbvhNode.bmaxz[i] = real(0);

            qbvhNode.bminx[i] = real(0);
            qbvhNode.bminy[i] = real(0);
            qbvhNode.bminz[i] = real(0);

            qbvhNode.leftChildrenIdx = 0;
        }

        qbvhNode.isLeaf = false;
    }

    int32_t StacklessQbvh::getChildren(
        std::vector<BvhNode>& listBvhNode,
        int32_t bvhNodeIdx,
        int32_t children[4])
    {
        const auto bvhNode = listBvhNode[bvhNodeIdx].node;

        // Invalidate children.
        children[0] = children[1] = children[2] = children[3] = -1;
        int32_t numChildren = 0;

        if (bvhNode->isLeaf()) {
            // No children.
            return numChildren;
        }

        const auto left = bvhNode->getLeft();

        if (left) {
            if (left->isLeaf()) {
                children[numChildren++] = left->getTraversalOrder();
            }
            else {
                const auto left_left = left->getLeft();
                const auto left_right = left->getRight();

                if (left_left) {
                    children[numChildren++] = left_left->getTraversalOrder();
                }
                if (left_right) {
                    children[numChildren++] = left_right->getTraversalOrder();
                }
            }
        }

        const auto right = bvhNode->getRight();

        if (right) {
            if (right->isLeaf()) {
                children[numChildren++] = right->getTraversalOrder();
            }
            else {
                const auto right_left = right->getLeft();
                const auto right_right = right->getRight();

                if (right_left) {
                    children[numChildren++] = right_left->getTraversalOrder();
                }
                if (right_right) {
                    children[numChildren++] = right_right->getTraversalOrder();
                }
            }
        }

        return numChildren;
    }

    inline int32_t intersectAABB(
        aten::vec4& result,
        const aten::ray& r,
        real t_min, real t_max,
        const aten::vec4& bminx, const aten::vec4& bmaxx,
        const aten::vec4& bminy, const aten::vec4& bmaxy,
        const aten::vec4& bminz, const aten::vec4& bmaxz)
    {
        // NOTE
        // No SSE...

        aten::vec3 invdir = real(1) / (r.dir + aten::vec3(real(1e-6)));
        aten::vec3 oxinvdir = -r.org * invdir;

        aten::vec4 invdx(invdir.x);
        aten::vec4 invdy(invdir.y);
        aten::vec4 invdz(invdir.z);

        aten::vec4 ox(oxinvdir.x);
        aten::vec4 oy(oxinvdir.y);
        aten::vec4 oz(oxinvdir.z);

        aten::vec4 minus_inf(-AT_MATH_INF);
        aten::vec4 plus_inf(AT_MATH_INF);

        // X
        auto fx = bmaxx * invdx + ox;
        auto nx = bminx * invdx + ox;

        // Y
        auto fy = bmaxy * invdy + oy;
        auto ny = bminy * invdy + oy;

        // Z
        auto fz = bmaxz * invdz + oz;
        auto nz = bminz * invdz + oz;

        auto tmaxX = max(fx, nx);
        auto tminX = min(fx, nx);

        auto tmaxY = max(fy, ny);
        auto tminY = min(fy, ny);

        auto tmaxZ = max(fz, nz);
        auto tminZ = min(fz, nz);

        auto t1 = min(min(tmaxX, tmaxY), min(tmaxZ, t_max));
        auto t0 = max(max(tminX, tminY), max(tminZ, t_min));

        int32_t ret = cmpLEQ(t0, t1);

        result.x = t0.x <= t1.x ? t0.x : AT_MATH_INF;
        result.y = t0.y <= t1.y ? t0.y : AT_MATH_INF;
        result.z = t0.z <= t1.z ? t0.z : AT_MATH_INF;
        result.w = t0.w <= t1.w ? t0.w : AT_MATH_INF;

        return ret;
    }

    // returns position of the rightmost set bit of n
    int32_t bitScan(int32_t n)
    {
        // NOTE
        // http://www.techiedelight.com/bit-hacks-part-3-playing-rightmost-set-bit-number/

        // if number is odd, return 1
        if (n & 1) {
            //return 1;
            return 0;
        }

        // unset rightmost bit and xor with number itself
        n = n ^ (n & (n - 1));

        // find the position of the only set bit in the result
        // we can directly return log2(n) + 1 from the function
#if 0
        int32_t pos = 0;
        while (n)
        {
            n = n >> 1;
            pos++;
        }
#else
        int32_t pos = (int32_t)(::log2(n) + 1);
#endif
        return pos - 1;
    }

    int32_t SkipCode(int32_t mask, int32_t pos)
    {
        return (((mask >> (pos + 1))) | (mask << (3 - pos))) & 7;
    }

    int32_t SkipCodeNext(int32_t code)
    {
        int32_t n = bitScan(code);
        int32_t newCode = code >> (n + 1);
        return newCode ^ code;
    }

    bool StacklessQbvh::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listQbvhNode, r, t_min, t_max, isect);
    }

    bool StacklessQbvh::hit(
        const context& ctxt,
        int32_t exid,
        const std::vector<std::vector<StacklessQbvhNode>>& listQbvhNode,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        int32_t nodeid = 0;
        uint32_t bitstack = 0;

        int32_t skipCode = 0;

        for (;;) {
            const StacklessQbvhNode* pnode = nullptr;

            if (nodeid >= 0) {
                pnode = &listQbvhNode[exid][nodeid];
            }

            if (!pnode) {
                break;
            }

            const auto numChildren = pnode->numChildren;

            if (pnode->isLeaf) {
                Intersection isectTmp;

                bool isHit = false;

                auto s = ctxt.getTransformable((int32_t)pnode->object_id);

                if (pnode->exid >= 0) {
                    // Traverse external StacklessQbvh.
                    const auto& param = s->getParam();

                    int32_t mtx_id = param.mtx_id;

                    aten::ray transformedRay;

                    if (mtx_id >= 0) {
                        const auto& mtxW2L = m_mtxs[mtx_id * 2 + 1];

                        transformedRay = mtxW2L.applyRay(r);
                    }
                    else {
                        transformedRay = r;
                    }

                    isHit = hit(
                        ctxt,
                        (int32_t)pnode->exid,
                        listQbvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else if (pnode->primid >= 0) {
                    auto f = ctxt.getTriangle((int32_t)pnode->primid);
                    isHit = f->hit(ctxt, r, t_min, t_max, isectTmp);

                    if (isHit) {
                        isectTmp.objid = s->id();
                    }
                }
                else {
                    // sphere, cube.
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
                // Hit test children aabb.
                aten::vec4 intersectT;
                auto hitMask = intersectAABB(
                    intersectT,
                    r,
                    t_min, t_max,
                    pnode->bminx, pnode->bmaxx,
                    pnode->bminy, pnode->bmaxy,
                    pnode->bminz, pnode->bmaxz);

                if (hitMask > 0) {
                    bitstack = bitstack << 3;

                    if (hitMask == 1) {
                        nodeid = (int32_t)pnode->leftChildrenIdx + 0;
                    }
                    else if (hitMask == 2) {
                        nodeid = (int32_t)pnode->leftChildrenIdx + 1;
                    }
                    else if (hitMask == 4) {
                        nodeid = (int32_t)pnode->leftChildrenIdx + 2;
                    }
                    else if (hitMask == 8) {
                        nodeid = (int32_t)pnode->leftChildrenIdx + 3;
                    }
                    else {
                        int32_t nearId_a = (intersectT.x < intersectT.y ? 0 : 1);
                        int32_t nearId_b = (intersectT.z < intersectT.w ? 2 : 3);

                        int32_t nearPos = (intersectT[nearId_a] < intersectT[nearId_b] ? nearId_a : nearId_b);

                        nodeid = (int32_t)pnode->leftChildrenIdx + nearPos;

                        skipCode = SkipCode(hitMask, nearPos);
                        bitstack = bitstack | skipCode;
                    }

                    continue;
                }
            }

            while ((skipCode = (bitstack & 7)) == 0) {
                if (bitstack == 0) {
                    return (isect.objid >= 0);
                }

                nodeid = (int32_t)pnode->parent;
                bitstack = bitstack >> 3;

                pnode = &listQbvhNode[exid][nodeid];
            }

            auto siblingPos = bitScan(skipCode);

            nodeid = (int32_t)pnode->sib[siblingPos];

            int32_t n = SkipCodeNext(skipCode);
            bitstack = bitstack ^ n;
        }

        return (isect.objid >= 0);
    }
}
