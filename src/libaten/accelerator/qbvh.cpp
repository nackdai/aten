#include "accelerator/qbvh.h"

//#pragma optimize( "", off)

namespace aten
{
    void qbvh::build(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
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

            listBvhNode.push_back(std::vector<BvhNode>());
            std::vector<accelerator*> dummy;

            registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1], dummy, nestedBvhMap);
            AT_ASSERT(dummy.empty());
        }

        m_listQbvhNode.resize(listBvhNode.size());

        // Convert to QBVH.
        for (int i = 0; i < listBvhNode.size(); i++) {
            bool isPrimitiveLeafBvh = (i > 0);

            auto numNodes = convertFromBvh(
                ctxt,
                isPrimitiveLeafBvh,
                listBvhNode[i],
                m_listQbvhNode[i]);
        }
    }

    void qbvh::registerBvhNodeToLinearList(
        bvhnode* root,
        bvhnode* parentNode,
        hitable* nestParent,
        const aten::mat4& mtxL2W,
        std::vector<BvhNode>& listBvhNode,
        std::vector<accelerator*>& listBvh,
        std::map<hitable*, accelerator*>& nestedBvhMap)
    {
        bvh::registerBvhNodeToLinearList<BvhNode>(
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
            [this](bvhnode* node, int exid, int subExid)
        {
            if (node->isLeaf()) {
                // NOTE
                // 0 はベースツリーなので、+1 する.
                node->setExternalId(exid);
            }
        });
    }

    uint32_t qbvh::convertFromBvh(
        const context& ctxt,
        bool isPrimitiveLeaf,
        std::vector<BvhNode>& listBvhNode,
        std::vector<QbvhNode>& listQbvhNode)
    {
        struct QbvhStackEntry {
            uint32_t qbvhNodeIdx;
            uint32_t bvhNodeIdx;

            QbvhStackEntry(uint32_t qbvh, uint32_t bvh)
                : qbvhNodeIdx(qbvh), bvhNodeIdx(bvh)
            {}
            QbvhStackEntry() {}
        };

        listQbvhNode.reserve(listBvhNode.size());
        listQbvhNode.push_back(QbvhNode());

        QbvhStackEntry stack[256];
        stack[0] = QbvhStackEntry(0, 0);

        int stackPos = 1;
        uint32_t numNodes = 1;

        int children[4];

        while (stackPos > 0) {
            auto top = stack[--stackPos];

            auto& qbvhNode = listQbvhNode[top.qbvhNodeIdx];
            const auto& bvhNode = listBvhNode[top.bvhNodeIdx];

            int numChildren = getChildren(listBvhNode, top.bvhNodeIdx, children);

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
            for (int i = 0; i < numChildren; i++) {
                stack[stackPos++] = QbvhStackEntry(numNodes, children[i]);

                listQbvhNode.push_back(QbvhNode());
                ++numNodes;
            }
        }

        return numNodes;
    }

    void qbvh::setQbvhNodeLeafParams(
        const context& ctxt,
        bool isPrimitiveLeaf,
        const BvhNode& bvhNode,
        QbvhNode& qbvhNode)
    {
        auto node = bvhNode.node;
        auto nestParent = bvhNode.nestParent;

        auto bbox = node->getBoundingbox();
        bbox = aten::aabb::transform(bbox, bvhNode.mtxL2W);

#if 0
        auto parent = node->getParent();
        qbvhNode.parent = (float)(parent ? parent->getTraversalOrder() : -1);
#endif

        qbvhNode.leftChildrenIdx = 0;

        qbvhNode.bmaxx.set(real(0));
        qbvhNode.bmaxy.set(real(0));
        qbvhNode.bmaxz.set(real(0));

        qbvhNode.bminx.set(real(0));
        qbvhNode.bminy.set(real(0));
        qbvhNode.bminz.set(real(0));

        qbvhNode.numChildren = 0;

        if (node->isLeaf()) {
#ifdef ENABLE_BVH_MULTI_TRIANGLES
            int numChildren = node->getChildrenNum();
            auto children = node->getChildren();

            qbvhNode.numChildren = numChildren;

            for (int i = 0; i < numChildren; i++) {
                auto ch = children[i];

                // 自分自身のIDを取得.
                qbvhNode.shapeidx[i] = (float)transformable::findIdxAsHitable(ch);

                // もしなかったら、ネストしているので親のIDを取得.
                if (qbvhNode.shapeidx[i] < 0) {
                    if (nestParent) {
                        qbvhNode.shapeid = (float)transformable::findIdxAsHitable(nestParent);
                        qbvhNode.shapeidx[i] = qbvhNode.shapeid;
                    }
                }

                // インスタンスの実体を取得.
                auto internalObj = ch->getHasObject();

                if (internalObj) {
                    ch = const_cast<hitable*>(internalObj);
                }

                if (isPrimitiveLeaf) {
                    // Leaves of this tree are primitive.
                    qbvhNode.primidx[i] = (float)face::findIdx(ch);

                    auto f = face::faces()[(int)qbvhNode.primidx[i]];

                    const auto& v0 = aten::VertexManager::getVertex(f->param.idx[0]);
                    const auto& v1 = aten::VertexManager::getVertex(f->param.idx[1]);

                    auto e1 = v1.pos - v0.pos;

                    qbvhNode.v0x[i] = v0.pos.x;
                    qbvhNode.v0y[i] = v0.pos.y;
                    qbvhNode.v0z[i] = v0.pos.z;

                    qbvhNode.e1x[i] = e1.x;
                    qbvhNode.e1y[i] = e1.y;
                    qbvhNode.e1z[i] = e1.z;

                    qbvhNode.exid = -1.0f;
                }
                else {
                    qbvhNode.shapeid = qbvhNode.shapeidx[i];
                    qbvhNode.exid = (float)node->getExternalId();
                }
            }
#else
            hitable* item = node->getItem();

            // 自分自身のIDを取得.
            qbvhNode.shapeid = (float)ctxt.findTransformableIdxFromPointer(item);

            // もしなかったら、ネストしているので親のIDを取得.
            if (qbvhNode.shapeid < 0) {
                if (nestParent) {
                    qbvhNode.shapeid = (float)ctxt.findTransformableIdxFromPointer(nestParent);
                }
            }

            // インスタンスの実体を取得.
            auto internalObj = item->getHasObject();

            if (internalObj) {
                item = const_cast<hitable*>(internalObj);
            }

            qbvhNode.meshid = (float)item->geomid();

            if (isPrimitiveLeaf) {
                // Leaves of this tree are primitive.
                qbvhNode.primid = (float)ctxt.findTriIdxFromPointer(item);
                qbvhNode.exid = -1.0f;
            }
            else {
                qbvhNode.exid = (float)node->getExternalId();
            }
#endif

            qbvhNode.isLeaf = (float)true;
        }
    }

    void qbvh::fillQbvhNode(
        QbvhNode& qbvhNode,
        std::vector<BvhNode>& listBvhNode,
        int children[4],
        int numChildren)
    {
        for (int i = 0; i < numChildren; i++) {
            int childIdx = children[i];
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
        for (int i = numChildren; i < 4; i++) {
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

    int qbvh::getChildren(
        std::vector<BvhNode>& listBvhNode,
        int bvhNodeIdx,
        int children[4])
    {
        const auto bvhNode = listBvhNode[bvhNodeIdx].node;

        // Invalidate children.
        children[0] = children[1] = children[2] = children[3] = -1;
        int numChildren = 0;

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

    inline int intersectAABB(
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

        int ret = cmpLEQ(t0, t1);
        result = t0;
        return ret;
    }

    inline int intersectTriangle(
        const context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        const float* primidx,
        const QbvhNode& qnode,
        aten::vec4& resultT,
        aten::vec4& resultA,
        aten::vec4& resultB)
    {
        // NOTE
        // https://github.com/githole/akari2/blob/master/qbvh.h

        // NOTE
        // No SSE...

        aten::vec4 v2x;
        aten::vec4 v2y;
        aten::vec4 v2z;

        for (int i = 0; i < qnode.numChildren; i++) {
            auto f = ctxt.getTriangle((int)primidx[i]);

            const auto& faceParam = f->getParam();

            const auto& v2 = ctxt.getVertex(faceParam.idx[2]);

            v2x[i] = v2.pos.x;
            v2y[i] = v2.pos.y;
            v2z[i] = v2.pos.z;
        }

        // e1 = v1 - v0
        const auto& e1_x = qnode.e1x;
        const auto& e1_y = qnode.e1y;
        const auto& e1_z = qnode.e1z;

        // e2 = v2 - v0
        auto e2_x = v2x - qnode.v0x;
        auto e2_y = v2y - qnode.v0y;
        auto e2_z = v2z - qnode.v0z;

        vec4 ox = vec4(r.org.x, r.org.x, r.org.x, r.org.x);
        vec4 oy = vec4(r.org.y, r.org.y, r.org.y, r.org.y);
        vec4 oz = vec4(r.org.z, r.org.z, r.org.z, r.org.z);

        // d
        vec4 d_x = vec4(r.dir.x, r.dir.x, r.dir.x, r.dir.x);
        vec4 d_y = vec4(r.dir.y, r.dir.y, r.dir.y, r.dir.y);
        vec4 d_z = vec4(r.dir.z, r.dir.z, r.dir.z, r.dir.z);

        // r = r.org - v0
        auto r_x = ox - qnode.v0x;
        auto r_y = oy - qnode.v0y;
        auto r_z = oz - qnode.v0z;

        // u = cross(d, e2)
        auto u_x = d_y * e2_z - d_z * e2_y;
        auto u_y = d_z * e2_x - d_x * e2_z;
        auto u_z = d_x * e2_y - d_y * e2_x;

        // v = cross(r, e1)
        auto v_x = r_y * e1_z - r_z * e1_y;
        auto v_y = r_z * e1_x - r_x * e1_z;
        auto v_z = r_x * e1_y - r_y * e1_x;

        // inv = real(1) / dot(u, e1)
        auto divisor = u_x * e1_x + u_y * e1_y + u_z * e1_z;
        auto inv = real(1) / (divisor + real(1e-6));

        // t = dot(v, e2) * inv
        auto t = (v_x * e2_x + v_y * e2_y + v_z * e2_z) * inv;

        // beta = dot(u, r) * inv
        auto beta = (u_x * r_x + u_y * r_y + u_z * r_z) * inv;

        // gamma = dot(v, d) * inv
        auto gamma = (v_x * d_x + v_y * d_y + v_z * d_z) * inv;

        resultT = t;
        resultA = beta;
        resultB = gamma;

        int res_b0 = cmpGEQ(beta, real(0));        // beta >= 0
        int res_b1 = cmpLEQ(beta, real(1));        // beta <= 1

        int res_g0 = cmpGEQ(gamma, real(0));    // gamma >= 0
        int res_g1 = cmpLEQ(gamma, real(1));    // gamma <= 1

        int res_bg1 = cmpLEQ(beta + gamma, real(1));    // beta + gammma <= 1

        int res_t0 = cmpGEQ(t, real(0));        // t >= 0

        int ret = res_b0 & res_b1 & res_g0 & res_g1 & res_bg1 & res_t0;

        return ret;
    }

    bool qbvh::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hit(ctxt, 0, m_listQbvhNode, r, t_min, t_max, isect);
    }

    bool qbvh::hit(
        const context& ctxt,
        int exid,
        const std::vector<std::vector<QbvhNode>>& listQbvhNode,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        static const uint32_t stacksize = 64;

        struct Intersect {
            const QbvhNode* node{ nullptr };
            real t;

            Intersect(const QbvhNode* n, real _t) : node(n), t(_t) {}
            Intersect() {}
        } stackbuf[stacksize];

        stackbuf[0] = Intersect(&listQbvhNode[exid][0], t_max);
        int stackpos = 1;

        while (stackpos > 0) {
            const auto& node = stackbuf[stackpos - 1];
            stackpos -= 1;

            if (node.t > t_max) {
                continue;
            }

            auto pnode = node.node;

            const auto numChildren = pnode->numChildren;

            if (pnode->isLeaf) {
                Intersection isectTmp;

                bool isHit = false;

#ifdef ENABLE_BVH_MULTI_TRIANGLES
                if (pnode->exid >= 0) {
                    // Traverse external qbvh.
                    auto s = shapes[(int)pnode->shapeid];
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
                        (int)pnode->exid,
                        listQbvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else {
                    aten::vec4 resultT, resultA, resultB;

                    int res = intersectTriangle(
                        resultT, resultA, resultB,
                        r,
                        t_min, t_max,
                        qnode->primidx,
                        *pnode);

                    for (int i = 0; i < pnode->numChildren; i++) {
                        if ((res & (1 << i)) && (resultT[i] < isectTmp.t)) {
                            isectTmp.t = resultT[i];
                            isectTmp.a = resultA[i];
                            isectTmp.b = resultB[i];

                            isectTmp.primid = (int)pnode->primidx[i];
                            isectTmp.objid = (int)pnode->shapeidx[i];

                            auto f = prims[isectTmp.primid];

                            isectTmp.mtrlid = f->param.mtrlid;
                            isectTmp.area = f->param.area;

                            isHit = true;
                        }
                    }

                    if (isHit) {
                        auto s = shapes[isectTmp.objid];
                        isectTmp.objid = s->id();
                    }
                }
#else
                auto s = ctxt.getTransformable((int)pnode->shapeid);

                if (pnode->exid >= 0) {
                    // Traverse external qbvh.
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
                        (int)pnode->exid,
                        listQbvhNode,
                        transformedRay,
                        t_min, t_max,
                        isectTmp);
                }
                else if (pnode->primid >= 0) {
                    auto f = ctxt.getTriangle((int)pnode->primid);
                    isHit = f->hit(ctxt, r, t_min, t_max, isectTmp);

                    if (isHit) {
                        isectTmp.objid = s->id();
                    }
                }
                else {
                    // sphere, cube.
                    isHit = s->hit(ctxt, r, t_min, t_max, isectTmp);
                }
#endif

                if (isHit) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;
                    }
                }
            }
            else {
                // Hit test children aabb.
                aten::vec4 interserctT;
                auto res = intersectAABB(
                    interserctT,
                    r,
                    t_min, t_max,
                    pnode->bminx, pnode->bmaxx,
                    pnode->bminy, pnode->bmaxy,
                    pnode->bminz, pnode->bmaxz);

                // Stack hit children.
                if (res > 0) {
                    for (int i = 0; i < numChildren; i++) {
                        if ((res & (1 << i)) > 0) {
                            stackbuf[stackpos] = Intersect(
                                &listQbvhNode[exid][(int)pnode->leftChildrenIdx + i],
                                interserctT[i]);
                            stackpos++;
                        }
                    }
                }
            }
        }

        return (isect.objid >= 0);
    }
}
