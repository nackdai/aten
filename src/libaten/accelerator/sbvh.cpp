#include <algorithm>
#include <iterator>
#include <numeric>

#include <omp.h>

#include "accelerator/sbvh.h"

//#pragma optimize( "", off)

// NOTE
// https://github.com/martinradev/BVH-algo-lib

namespace aten
{
    // Check if the surface area of merged AABBs is bigger than the specified cost.
    inline bool checkAABBOverlap(
        const aabb& box0,
        const aabb& box1,
        real& overlap)
    {
        aabb delta = aabb(
            aten::max(box0.minPos(), box1.minPos()),
            aten::min(box0.maxPos(), box1.maxPos()));

        if (delta.isValid()) {
            overlap = delta.computeSurfaceArea();
            return true;
        }

        return false;
    }

    template <class Iter, class Cmp>
    void mergeSort(Iter first, Iter last, Cmp cmp)
    {
        const auto numThreads = ::omp_get_max_threads();

        const auto numItems = last - first;
        const auto numItemPerThread = numItems / numThreads;

        // Sort each blocks.
#pragma omp parallel num_threads(numThreads)
        {
            const auto idx = ::omp_get_thread_num();
        
            const auto startPos = idx * numItemPerThread;
            const auto endPos = (idx + 1 == numThreads ? numItems : startPos + numItemPerThread);

            std::sort(first + startPos, first + endPos, cmp);
        }

        int blockNum = numThreads;

        // Merge blocks.
        while (blockNum >= 2)
        {
            const auto numItemsInBlocks = numItems / blockNum;

            // ２つのブロックを１つにマージする.
#pragma omp parallel num_threads(blockNum / 2)
            {
                const auto idx = ::omp_get_thread_num();

                // ２つのブロックを１つにマージするので、numItemsInBlocks * 2 となる.
                auto startPos = idx * numItemsInBlocks * 2;

                // 中間地点なので、numItemsInBlocks * 2 の半分で numItemsInBlocks となる.
                auto pivot = startPos + numItemsInBlocks;

                // 終端.
                auto endPos = (idx + 1 == blockNum / 2 ? numItems : pivot + numItemsInBlocks);

                std::inplace_merge(first + startPos, first + pivot, first + endPos);
            }

            // ２つを１つにマージするので、半分になる.
            blockNum = (blockNum + 1) / 2;
        }
    }

    void sbvh::build(
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

    void sbvh::buildAsNestedTree(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        AT_ASSERT(m_isNested);

        if (m_threadedNodes.size() > 0) {
            // Imported tree already.

            // Search offset triangle index.
            m_offsetTriIdx = INT32_MAX;

            for (uint32_t i = 0; i < num; i++) {
                auto tri = (face*)list[i];
                m_offsetTriIdx = std::min<uint32_t>(m_offsetTriIdx, tri->getId());
            }

            // Offset triangle index.
#pragma omp parallel for
            for (int i = 0; i < m_threadedNodes[0].size(); i++) {
                auto& node = m_threadedNodes[0][i];
                if (node.isLeaf()) {
                    node.triid += m_offsetTriIdx;
                }
            }
        }
        else {
            onBuild(ctxt, list, num);

            makeTreelet();
        }
    }

    void sbvh::buildAsTopLayerTree(
        const context& ctxt,
        hitable** list,
        uint32_t num,
        aabb* bbox)
    {
        AT_ASSERT(!m_isNested);

        // Tell not to build bottome layer.
        m_bvh.disableLayer();

        // Build top layer bvh.
        m_bvh.build(ctxt, list, num, bbox);

        auto boundingBox = m_bvh.getBoundingbox();

        const auto& nestedBvh = m_bvh.getNestedAccel();

        // NOTE
        // GPGPU処理用に threaded bvh(top layer) と sbvh を同じメモリ空間上に格納するため、１つのリストで管理する.
        // そのため、+1する.
        m_threadedNodes.resize(nestedBvh.size() + 1);

        // Copy top layer bvh nodes to the array which SBVH has.
        const auto& toplayer = m_bvh.getNodes()[0];
        m_threadedNodes[0].resize(toplayer.size());
        memcpy(&m_threadedNodes[0][0], &toplayer[0], toplayer.size() * sizeof(ThreadedSbvhNode));

        // Convert to threaded bvh.
        for (int i = 0; i < nestedBvh.size(); i++) {
            auto accel = nestedBvh[i];

            auto box = accel->getBoundingbox();
            boundingBox.expand(box);

            if (accel->getAccelType() == AccelType::Sbvh) {
                // TODO
                auto bvh = (sbvh*)nestedBvh[i];

                bvh->buildVoxel(ctxt);

                std::vector<int> indices;
                bvh->convert(
                    m_threadedNodes[i + 1],
                    (int)m_refIndices.size(),
                    indices);

                m_refIndices.insert(m_refIndices.end(), indices.begin(), indices.end());
            }
        }

        setBoundingBox(boundingBox);
    }

    void sbvh::onBuild(
        const context& ctxt,
        hitable** list,
        uint32_t num)
    {
        std::vector<face*> tris;
        tris.reserve(num);

        for (uint32_t i = 0; i < num; i++) {
            tris.push_back((face*)list[i]);
        }

        aabb rootBox;

        m_refs.clear();
        m_refs.reserve(2 * tris.size());
        m_refs.resize(tris.size());

        m_offsetTriIdx = INT32_MAX;

        for (uint32_t i = 0; i < tris.size(); i++) {
            m_refs[i].triid = i;
            m_refs[i].bbox = tris[i]->computeAABB(ctxt);

            m_offsetTriIdx = std::min<int>(m_offsetTriIdx, tris[i]->getId());

            rootBox.expand(m_refs[i].bbox);
        }

        if (isExporting()) {
            // No offset triangle index for exporting sbvh data which are multiple in a obj.
            m_offsetTriIdx = 0;
        }

        // Set as bounding box.
        setBoundingBox(rootBox);

        // Referenceのインデックスリストを作成.
        std::vector<uint32_t> refIndices(m_refs.size());
        std::iota(refIndices.begin(), refIndices.end(), 0);

        auto rootSurfaceArea = rootBox.computeSurfaceArea();

        // TODO
        const real areaAlpha = real(1e-5);

        struct SBVHEntry {
            SBVHEntry() {}
            SBVHEntry(uint32_t id, uint32_t d) : nodeIdx(id), depth(d) {}

            uint32_t nodeIdx;
            uint32_t depth;
        } stack[128];

        int stackpos = 1;
        stack[0] = SBVHEntry(0, 0);

        m_nodes.reserve(m_refs.size() * 3);
        m_nodes.push_back(SBVHNode());
        m_nodes[0] = SBVHNode(std::move(refIndices), rootBox);

        uint32_t numNodes = 1;

        m_refIndexNum = 0;

        m_maxDepth = 0;

        while (stackpos > 0)
        {
            auto top = stack[--stackpos];
            auto& node = m_nodes[top.nodeIdx];

            m_maxDepth = std::max<uint32_t>(node.depth, m_maxDepth);

            // enough triangles so far.
            if (node.refIds.size() <= m_maxTriangles) {
                m_refIndexNum += (uint32_t)node.refIds.size();
                continue;
            }

            real objCost = 0.0f;
            aabb objLeftBB;
            aabb objRightBB;
            int sahBin = -1;
            int sahComponent = -1;
            findObjectSplit(node, objCost, objLeftBB, objRightBB, sahBin, sahComponent);

            // check whether the object split produces overlapping nodes
            // if so, check whether we are close enough to the root so that the spatial split makes sense.
            real overlapCost = real(0);
            bool needComputeSpatial = false;

            // Check if the surface area of merged AABBs is bigger than the specified cost.
            bool isOverrlap = checkAABBOverlap(objLeftBB, objRightBB, overlapCost);

            if (isOverrlap) {
                // If the AABBs are overlapped, check whether the node is needed to split.
                needComputeSpatial = (overlapCost / rootSurfaceArea) >= areaAlpha;
            }

            float spatialCost = AT_MATH_INF;
            aabb spatialLeftBB;
            aabb spatialRightBB;
            real spatialSplitPlane = real(0);
            int spatialDimension = -1;
            int leftCnt = 0;
            int rightCnt = 0;

            if (needComputeSpatial) {
                findSpatialSplit(
                    node,
                    spatialCost,
                    leftCnt, rightCnt,
                    spatialLeftBB, spatialRightBB,
                    spatialDimension,
                    spatialSplitPlane);
            }

            std::vector<uint32_t> leftList;
            std::vector<uint32_t> rightList;

            // if we have compute the spatial cost and it is better than the binned sah cost,
            // then do the split.

            int usedAxis = 0;

            if (needComputeSpatial && spatialCost <= objCost) {
                // use spatial split.
                spatialSort(
                    node,
                    spatialSplitPlane,
                    spatialDimension,
                    spatialCost,
                    leftCnt, rightCnt,
                    spatialLeftBB, spatialRightBB,
                    leftList, rightList);

                objLeftBB = spatialLeftBB;
                objRightBB = spatialRightBB;

                usedAxis = spatialDimension;
            }
            else {
                // use object split.

                // check whether the binned sah has failed.
                // if so, we have to do the object median split.
                // it happens for some scenes, but it happens very close to the leaves.
                // i.e. when we are left with 8-16 tightly packed references which end up in the same bin.
                if (sahBin == -1) {
                    auto axisDelta = node.bbox.size();

                    auto maxVal = std::max(std::max(axisDelta.x, axisDelta.y), axisDelta.z);

                    int bestAxis = (maxVal == axisDelta.x
                        ? 0
                        : maxVal == axisDelta.y ? 1 : 2);

                    usedAxis = bestAxis;

                    // bestAxisに基づいてbboxの位置に応じてソート.
                    mergeSort(
                        node.refIds.begin(),
                        node.refIds.end(),
                        [bestAxis, this](const uint32_t a, const uint32_t b) {
                        return m_refs[a].bbox.getCenter()[bestAxis] < m_refs[b].bbox.getCenter()[bestAxis];
                    });

                    // 分割AABBの大きさをリセット.
                    objLeftBB.empty();
                    objRightBB.empty();

                    // distribute in left and right child evenly.
                    // 半分ずつ右と左に均等に分割.
                    for (int i = 0; i < node.refIds.size(); i++) {
                        const auto id = node.refIds[i];
                        const auto& ref = m_refs[id];

                        if (i < node.refIds.size() / 2) {
                            leftList.push_back(node.refIds[i]);
                            objLeftBB.expand(ref.bbox);
                        }
                        else {
                            rightList.push_back(node.refIds[i]);
                            objRightBB.expand(ref.bbox);
                        }
                    }
                }
                else {
                    objectSort(
                        node,
                        sahBin,
                        sahComponent,
                        leftList, rightList);

                    usedAxis = sahComponent;
                }
            }

            // push left and right.

            auto leftIdx = numNodes;
            auto rightIdx = leftIdx + 1;

            AT_ASSERT(leftList.size() + rightList.size() >= node.refIds.size());

            // dont with this object, deallocate memory for the current node.
            node.refIds.clear();
            node.setChild(leftIdx, rightIdx);

            // copy node data to left and right children.
            // ここで push_back することで、std::vector 内部のメモリ構造が変わることがあるので、参照である node の変更はこの前までに終わらせること.
            m_nodes.push_back(SBVHNode());
            m_nodes.push_back(SBVHNode());

            m_nodes[leftIdx] = SBVHNode(std::move(leftList), objLeftBB);
            m_nodes[rightIdx] = SBVHNode(std::move(rightList), objRightBB);

            m_nodes[leftIdx].parent = top.nodeIdx;
            m_nodes[leftIdx].depth = top.depth + 1;

            m_nodes[rightIdx].parent = top.nodeIdx;
            m_nodes[rightIdx].depth = top.depth + 1;

            stack[stackpos++] = SBVHEntry(leftIdx, top.depth + 1);
            stack[stackpos++] = SBVHEntry(rightIdx, top.depth + 1);

            numNodes += 2;
        }

        AT_ASSERT(m_nodes.size() == numNodes);
    }

    inline real evalPreSplitCost(
        real leftBoxArea, int numLeft,
        real rightBoxArea, int numRight)
    {
        return leftBoxArea * numLeft + rightBoxArea * numRight;
    }

    void sbvh::findObjectSplit(
        SBVHNode& node,
        real& cost,
        aabb& leftBB,
        aabb& rightBB,
        int& splitBinPos,
        int& axis)
    {
        std::vector<Bin> bins(m_numBins);

        aabb bbCentroid = aabb(node.bbox.maxPos(), node.bbox.minPos());

        uint32_t refNum = (uint32_t)node.refIds.size();

        // compute the aabb of all centroids.
        for (uint32_t i = 0; i < refNum; i++) {
            auto id = node.refIds[i];
            const auto& ref = m_refs[id];
            auto center = ref.bbox.getCenter();
            bbCentroid.expand(center);
        }

        cost = AT_MATH_INF;
        splitBinPos = -1;
        axis = -1;

        auto centroidMin = bbCentroid.minPos();
        auto centroidMax = bbCentroid.maxPos();

        // for each dimension check the best splits.
        for (int dim = 0; dim < 3; ++dim)
        {
            // Skip empty axis.
            if ((centroidMax[dim] - centroidMin[dim]) == 0.0f) {
                continue;
            }

            const real invLen = real(1) / (centroidMax[dim] - centroidMin[dim]);

            // clear bins;
            for (uint32_t i = 0; i < m_numBins; i++) {
                bins[i] = Bin();
            }

            // distribute references in the bins based on the centroids.
            for (uint32_t i = 0; i < refNum; i++) {
                // ノードに含まれる三角形のAABBについて計算.

                auto id = node.refIds[i];
                const auto& ref = m_refs[id];

                auto center = ref.bbox.getCenter();

                // 分割情報(bins)へのインデックス.
                // 最小端点から三角形AABBの中心への距離を軸の長さで正規化することで計算.
                int binIdx = (int)(m_numBins * ((center[dim] - centroidMin[dim]) * invLen));

                binIdx = std::min<int>(binIdx, m_numBins - 1);
                AT_ASSERT(binIdx >= 0);

                bins[binIdx].start += 1;    // Binに含まれる三角形の数を増やす.
                bins[binIdx].bbox.expand(ref.bbox);
            }

            // 後ろから分割情報を蓄積する.
            bins[m_numBins - 1].accum = bins[m_numBins - 1].bbox;
            bins[m_numBins - 1].end = bins[m_numBins - 1].start;

            for (int i = m_numBins - 2; i >= 0; i--) {
                // ここまでのAABB全体サイズ.
                bins[i].accum = bins[i + 1].accum;

                // 自分自身のAABBを含める.
                bins[i].accum.expand(bins[i].bbox);

                // ここまでの三角形数.
                bins[i].end = bins[i].start + bins[i + 1].end;
            }

            // keep only one variable for forward accumulation
            auto leftBox = bins[0].bbox;

            // find split.
            for (uint32_t i = 0; i < m_numBins - 1; i++) {
                // 一つずつ範囲を広げていく.
                leftBox.expand(bins[i].bbox);

                auto leftArea = leftBox.computeSurfaceArea();

                // i から先の全体範囲が右側になる.
                auto rightArea = bins[i + 1].accum.computeSurfaceArea();

                int rightCount = bins[i + 1].end;
                int leftCount = refNum - rightCount;

                if (leftCount == 0) {
                    continue;
                }

                // Evaluate split cost.
                auto splitCost = evalPreSplitCost(leftArea, leftCount, rightArea, rightCount);

                // If the split cost is lower than the specified cost, update cost, bin position, axis etc.
                if (splitCost < cost)
                {
                    cost = splitCost;
                    splitBinPos = i;
                    axis = dim;

                    leftBB = leftBox;

                    // i から先の全体範囲が右側になる.
                    rightBB = bins[i + 1].accum;
                }
            }
        }
    }

    void sbvh::findSpatialSplit(
        SBVHNode& node,
        real& cost,
        int& retLeftCount,
        int& retRightCount,
        aabb& leftBB,
        aabb& rightBB,
        int& bestAxis,
        real& splitPlane)
    {
        cost = AT_MATH_INF;
        splitPlane = -1;
        bestAxis = -1;

        std::vector<Bin> bins(m_numBins);

        uint32_t refNum = (uint32_t)node.refIds.size();

        const auto& box = node.bbox;
        const auto boxMin = box.minPos();
        const auto boxMax = box.maxPos();

        // check along each dimension
        for (int dim = 0; dim < 3; ++dim)
        {
            const auto segmentLength = boxMax[dim] - boxMin[dim];

            if (segmentLength == real(0)) {
                continue;
            }

            const auto invLen = real(1) / segmentLength;

            // 分割情報当たりの軸の長さ.
            const auto lenghthPerBin = segmentLength / (float)m_numBins;

            // clear bins;
            for (uint32_t i = 0; i < m_numBins; i++) {
                bins[i] = Bin();
            }

            // Check all triangles which the node has.
            for (uint32_t i = 0; i < refNum; i++) {
                const auto id = node.refIds[i];
                const auto& ref = m_refs[id];

                const auto triMin = ref.bbox.minPos()[dim];
                const auto triMax = ref.bbox.maxPos()[dim];

                // split each triangle into references.
                // each triangle will be recorded into multiple bins.
                // 三角形が入る分割情報のインデックス範囲を計算.
                int binStartIdx = (int)(m_numBins * ((triMin - boxMin[dim]) * invLen));
                int binEndIdx = (int)(m_numBins * ((triMax - boxMin[dim]) * invLen));

                binStartIdx = aten::clamp<int>(binStartIdx, 0, m_numBins - 1);
                binEndIdx = aten::clamp<int>(binEndIdx, 0, m_numBins - 1);

                //AT_ASSERT(binStartIdx <= binEndIdx);

                for (int n = binStartIdx; n <= binEndIdx; n++) {
                    const auto binMin = boxMin[dim] + n * lenghthPerBin;
                    const auto binMax = boxMin[dim] + (n + 1) * lenghthPerBin;
                    AT_ASSERT(binMin <= binMax);

                    bins[n].bbox.expand(ref.bbox);

                    if (bins[n].bbox.minPos()[dim] < binMin) {
                        bins[n].bbox.minPos()[dim] = binMin;
                    }

                    if (bins[n].bbox.maxPos()[dim] > binMax) {
                        bins[n].bbox.maxPos()[dim] = binMax;
                    }
                }

                // 分割情報が取り扱う三角形数を更新.
                bins[binStartIdx].start++;
                bins[binEndIdx].end++;
            }

            // augment the bins from right to left.

            bins[m_numBins - 1].accum = bins[m_numBins - 1].bbox;

            for (int n = m_numBins - 2; n >= 0; n--) {
                // ここまでのAABB全体サイズ.
                bins[n].accum = bins[n + 1].accum;

                // 自分自身のAABBを含める.
                bins[n].accum.expand(bins[n].bbox);

                // ここまでの三角形数.
                bins[n].end += bins[n + 1].end;
            }

            int leftCount = 0;
            auto leftBox = bins[0].bbox;

            // find split.
            for (uint32_t n = 0; n < m_numBins - 1; n++) {
                const int rightCount = bins[n + 1].end;
                leftCount += bins[n].start;

                leftBox.expand(bins[n].bbox);

                AT_ASSERT(leftBox.maxPos()[dim] <= bins[n + 1].bbox.minPos()[dim]);

                real leftArea = leftBox.computeSurfaceArea();
                real rightArea = bins[n + 1].accum.computeSurfaceArea();

                const real splitCost = evalPreSplitCost(leftArea, leftCount, rightArea, rightCount);

                if (splitCost < cost) {
                    cost = splitCost;

                    leftBB = leftBox;
                    rightBB = bins[n + 1].accum;

                    bestAxis = dim;
                    splitPlane = bins[n + 1].accum.minPos()[dim];

                    retLeftCount = leftCount;
                    retRightCount = rightCount;
                }
            }
        }
    }

    void sbvh::spatialSort(
        SBVHNode& node,
        real splitPlane,
        int axis,
        real splitCost,
        int leftCnt,
        int rightCnt,
        aabb& leftBB,
        aabb& rightBB,
        std::vector<uint32_t>& leftList,
        std::vector<uint32_t>& rightList)
    {
        std::vector<Bin> bins(m_numBins);

        real rightSurfaceArea = rightBB.computeSurfaceArea();
        real leftSurfaceArea = leftBB.computeSurfaceArea();

        uint32_t refNum = (uint32_t)node.refIds.size();

        // distribute the refenreces to left, right or both children.
        for (uint32_t i = 0; i < refNum; i++) {
            const auto refIdx = node.refIds[i];
            const auto& ref = m_refs[refIdx];

            const auto refMin = ref.bbox.minPos()[axis];
            const auto refMax = ref.bbox.maxPos()[axis];

            if (refMax <= splitPlane) {
                // 分割位置より左.
                leftList.push_back(refIdx);
            }
            else if (refMin >= splitPlane) {
                // 分割位置より右.
                rightList.push_back(refIdx);
            }
            else {
                // split the reference.

                // check possible unsplit.

                // NOTE
                // 論文より.
                // Csplit = SA(B1) * N1 + SA(B2) * N2
                //     C1 = SA(B1 ∪ B△) * N1 + SA(B2) * (N2 - 1)
                //     C2 = SA(B1) * (N1 - 1) + SA(B2 ∪ B△) * N2

                aabb leftUnsplitBB = leftBB;
                leftUnsplitBB.expand(ref.bbox);
                const real leftUnsplitBBArea = leftUnsplitBB.computeSurfaceArea();
                const real unsplitLeftCost = leftUnsplitBBArea * leftCnt + rightSurfaceArea * (rightCnt - 1);

                aabb rightUnsplitBB = rightBB;
                rightUnsplitBB.expand(ref.bbox);
                const real rightUnsplitBBArea = rightUnsplitBB.computeSurfaceArea();
                const real unsplitRightCost = leftSurfaceArea * (leftCnt - 1) + rightUnsplitBBArea * rightCnt;

                if (unsplitLeftCost < splitCost && unsplitLeftCost <= unsplitRightCost) {
                    // put only into left only.
                    leftList.push_back(refIdx);

                    // update params.
                    leftSurfaceArea = leftUnsplitBBArea;
                    leftBB = leftUnsplitBB;
                    rightCnt -= 1;
                    splitCost = unsplitLeftCost;
                }
                else if (unsplitRightCost <= unsplitLeftCost && unsplitRightCost < splitCost) {
                    // put only into right only.
                    rightList.push_back(refIdx);

                    rightSurfaceArea = leftUnsplitBBArea;
                    rightBB = rightUnsplitBB;
                    leftCnt -= 1;
                    splitCost = unsplitRightCost;
                }
                else {
                    // push left and right.
                    // 二つに分割.

                    Reference leftRef(m_refs[refIdx]);

                    // バウンディングボックスを更新.
                    if (leftRef.bbox.maxPos()[axis] > splitPlane) {
                        leftRef.bbox.maxPos()[axis] = splitPlane;
                    }

                    Reference rightRef(m_refs[refIdx]);

                    // バウンディングボックスを更新.
                    if (rightRef.bbox.minPos()[axis] < splitPlane) {
                        rightRef.bbox.minPos()[axis] = splitPlane;
                    }

                    m_refs[refIdx] = leftRef;
                    m_refs.push_back(rightRef);

                    leftList.push_back(refIdx);
                    rightList.push_back((uint32_t)m_refs.size() - 1);
                }
            }
        }
    }

    void sbvh::objectSort(
        SBVHNode& node,
        int splitBin,
        int axis,
        std::vector<uint32_t>& leftList,
        std::vector<uint32_t>& rightList)
    {
        std::vector<Bin> bins(m_numBins);

        aabb bbCentroid = aabb(node.bbox.maxPos(), node.bbox.minPos());

        const auto refNum = node.refIds.size();

        // compute the aabb of all centroids.
        for (int i = 0; i < refNum; i++) {
            const auto id = node.refIds[i];
            const auto& ref = m_refs[id];
            auto centroid = ref.bbox.getCenter();
            bbCentroid.expand(centroid);
        }

        const auto invLen = real(1) / (bbCentroid.maxPos()[axis] - bbCentroid.minPos()[axis]);

        // distribute to left and right based on the provided split bin
        for (int i = 0; i < refNum; i++) {
            const auto id = node.refIds[i];
            const auto& ref = m_refs[id];

            auto center = ref.bbox.getCenter();

            // 分割情報(bins)へのインデックス.
            // 最小端点から三角形AABBの中心への距離を軸の長さで正規化することで計算.
            int binIdx = (int)(m_numBins * ((center[axis] - bbCentroid.minPos()[axis]) * invLen));

            binIdx = aten::clamp<int>(binIdx, 0, m_numBins - 1);

            if (binIdx <= splitBin) {
                // bin indexがsplitBinより小さいので左.
                leftList.push_back(id);
            }
            else {
                // bin indexがsplitBinより大きいので右.
                rightList.push_back(id);
            }
        }
    }

    void sbvh::convert(
        std::vector<ThreadedSbvhNode>& nodes,
        int offset,
        std::vector<int>& indices) const
    {
        if (m_threadedNodes.size() > 0
            && m_threadedNodes[0].size() > 0)
        {
            // Imported noddes already. So, just copy.
            std::copy(
                m_threadedNodes[0].begin(),
                m_threadedNodes[0].end(),
                std::back_inserter(nodes));
            return;
        }

        indices.resize(m_refIndexNum);
        nodes.resize(m_nodes.size());

        // in order traversal to index nodes
        std::vector<int> inOrderIndices;
        getOrderIndex(inOrderIndices);

        struct ThreadedEntry {
            ThreadedEntry() {}

            ThreadedEntry(int idx, int _parentSiblind)
                : nodeIdx(idx), parentSibling(_parentSiblind)
            {}

            int nodeIdx{ -1 };
            int parentSibling{ -1 };
        } stack[128];

        int stackpos = 1;

        int refIndicesCount = 0;
        int nodeCount = 0;

        stack[0] = ThreadedEntry(0, -1);

        while (stackpos > 0) {
            auto entry = stack[stackpos - 1];
            stackpos -= 1;

            const auto& sbvhNode = m_nodes[entry.nodeIdx];
            auto& thrededNode = nodes[entry.nodeIdx];

            thrededNode.boxmin = sbvhNode.bbox.minPos();
            thrededNode.boxmax = sbvhNode.bbox.maxPos();

            if (nodeCount + 1 == nodes.size()) {
                thrededNode.hit = -1.0f;
            }
            else {
                thrededNode.hit = (float)inOrderIndices[nodeCount + 1];
            }

            if (sbvhNode.isLeaf()) {
                if (nodeCount + 1 == nodes.size()) {
                    thrededNode.miss = -1.0f;
                }
                else {
                    thrededNode.miss = (float)inOrderIndices[nodeCount + 1];
                }

#if (SBVH_TRIANGLE_NUM == 1)
                const auto refid = sbvhNode.refIds[0];
                const auto& ref = m_refs[refid];
                thrededNode.triid = (float)(ref.triid + m_offsetTriIdx);
                thrededNode.isleaf = 1;
#else
                thrededNode.refIdListStart = (float)refIndicesCount + offset;
                thrededNode.refIdListEnd = thrededNode.refIdListStart + (float)sbvhNode.refIds.size();
#endif

                // 参照する三角形インデックスを配列に格納.
                // 分割しているので、重複する場合もあるので、別配列に格納していく.
                for (int i = 0; i < sbvhNode.refIds.size(); i++) {
                    const auto refId = sbvhNode.refIds[i];
                    const auto& ref = m_refs[refId];

                    indices[refIndicesCount++] = ref.triid + m_offsetTriIdx;
                }
            }
            else {
#if (SBVH_TRIANGLE_NUM == 1)
                thrededNode.triid = -1;
                thrededNode.isleaf = -1;
#else
                thrededNode.refIdListStart = -1;
                thrededNode.refIdListEnd = -1;
#endif

                thrededNode.miss = (float)entry.parentSibling;

                stack[stackpos++] = ThreadedEntry(sbvhNode.right, entry.parentSibling);
                stack[stackpos++] = ThreadedEntry(sbvhNode.left, sbvhNode.right);

                // For voxel.
                {
                    thrededNode.voxeldepth = AT_DISABLE_VOXEL;

                    if (sbvhNode.isTreeletRoot) {
                        const auto& found = m_treelets.find(entry.nodeIdx);
                        if (found != m_treelets.end()) {
                            const auto& treelet = found->second;

                            if (treelet.enabled) {
                                int32_t depth = sbvhNode.depth;

                                thrededNode.voxeldepth = AT_SET_VOXEL_DETPH(depth);

                                thrededNode.mtrlid = (float)treelet.mtrlid;
                            }
                        }
                    }
                }
            }

            nodeCount++;
        }

    }

    void sbvh::getOrderIndex(std::vector<int>& indices) const
    {
        // Traverse the tree and register index to the list.

        indices.reserve(m_nodes.size());

        int stack[128] = { 0 };

        int stackpos = 1;

        while (stackpos > 0) {
            int idx = stack[stackpos - 1];
            stackpos -= 1;

            const auto& sbvhNode = m_nodes[idx];

            indices.push_back(idx);

            if (!sbvhNode.isLeaf()) {
                stack[stackpos++] = sbvhNode.right;
                stack[stackpos++] = sbvhNode.left;
            }
        }
    }

    bool sbvh::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
        return hit(ctxt, r, t_min, t_max, false, isect);
    }

    bool sbvh::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        bool enableLod,
        Intersection& isect) const
    {
        auto& mtxs = m_bvh.getMatrices();

        const auto& topLayerBvhNode = m_bvh.getNodes()[0];

        real hitt = AT_MATH_INF;

        int nodeid = 0;

        for (;;) {
            const ThreadedBvhNode* node = nullptr;

            if (nodeid >= 0) {
                //node = &topLayerBvhNode[nodeid];
                node = (const ThreadedBvhNode*)&m_threadedNodes[0][nodeid];
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
                        const auto& mtxW2L = mtxs[mtxid * 2 + 1];

                        transformedRay = mtxW2L.applyRay(r);
                    }
                    else {
                        transformedRay = r;
                    }

                    //int exid = node->mainExid;
                    int exid = *(int*)(&node->exid);
                    bool hasLod = AT_BVHNODE_HAS_LOD(exid);
                    exid = hasLod && enableLod ? AT_BVHNODE_LOD_EXID(exid) : AT_BVHNODE_MAIN_EXID(exid);
                    //exid = AT_BVHNODE_LOD_EXID(exid);

                    isHit = hit(
                        ctxt,
                        exid,
                        transformedRay,
                        t_min, t_max,
                        isectTmp,
                        enableLod);
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
                        isect.objid = s->id();
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

    bool sbvh::hit(
        const context& ctxt,
        int exid,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect,
        bool enableLod) const
    {
        real hitt = AT_MATH_INF;

        int nodeid = 0;

        for (;;) {
            const ThreadedSbvhNode* node = nullptr;

            if (nodeid >= 0) {
                node = &m_threadedNodes[exid][nodeid];
            }

            if (!node) {
                break;
            }

            bool isHit = false;

            if (node->isLeaf()) {
                Intersection isectTmp;

#if (SBVH_TRIANGLE_NUM == 1)
                auto prim = ctxt.getTriangle((int)node->triid);
                isHit = prim->hit(ctxt, r, t_min, t_max, isectTmp);

                if (isHit) {
                    const auto& primParam = prim->getParam();
                    isectTmp.meshid = primParam.gemoid;
                }
#else
                int start = (int)node->refIdListStart;
                int end = (int)node->refIdListEnd;

                auto tmpTmax = t_max;

                for (int i = start; i < end; i++) {
                    int triid = m_refIndices[i];

                    auto prim = prims[triid];
                    auto hit = prim->hit(r, t_min, tmpTmax, isectTmp);

                    if (hit) {
                        isHit = true;
                        tmpTmax = isectTmp.t;
                        isectTmp.meshid = prim->param.gemoid;
                    }
                }
#endif

                if (isHit) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;
                    }
                }
            }
#if 1
            else if (enableLod && AT_IS_VOXEL(node->voxeldepth))
            {
                int voxeldepth = AT_GET_VOXEL_DEPTH(node->voxeldepth);

                float t_result = 0.0f;
                aten::vec3 nml;
                isHit = aten::aabb::hit(r, node->boxmin, node->boxmax, t_min, t_max, t_result, nml);

                // TODO
                // Fixed depth for debug...
                if (isHit && voxeldepth == 3) {
                    Intersection isectTmp;

                    isectTmp.isVoxel = true;

                    // TODO
                    // L2Wマトリクス.

                    isectTmp.t = t_result;

                    isectTmp.nml_x = nml.x;
                    isectTmp.nml_y = nml.y;
                    isectTmp.nml_z = nml.z;

                    isectTmp.mtrlid = (short)node->mtrlid;

                    // Dummy value, return ray hit voxel.
                    isectTmp.objid = 1;

                    // LODにヒットしたので、子供（詳細）は探索しないようにする.
                    isHit = false;

                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;
                    }
                }
            }
#endif
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

    struct SbvhFileHeader {
        char magic[4];

        union {
            uint32_t v;
            uint8_t version[4];
        };

        uint32_t nodeNum;
        uint32_t maxDepth;

        uint32_t cntMtrlForVoxel;

        float boxmin[3];
        float boxmax[3];
    };

    bool sbvh::exportTree(
        const context& ctxt,
        const char* path)
    {
        m_threadedNodes.resize(1);

        // Build voxel.
        if (!m_treelets.empty() && !m_nodes.empty()) {
            buildVoxel(ctxt);
        }

        std::vector<int> indices;
        convert(
            m_threadedNodes[0],
            0,
            indices);

        FILE* fp = fopen(path, "wb");
        if (!fp) {
            // TODO
            // throw exception...
            AT_ASSERT(false);
            return false;
        }

        // Gather material information.
        std::map<int, std::string> mtrlMap;
        {
            for (auto it : m_treelets) {
                const auto& treelet = it.second;
                if (treelet.mtrlid >= 0) {
                    const auto mtrl = ctxt.getMaterial(treelet.mtrlid);
                    AT_ASSERT(mtrl);

                    mtrlMap.insert(std::make_pair(treelet.mtrlid, mtrl->nameString()));
                }
            }
        }

        SbvhFileHeader header;
        {
            header.magic[0] = 'S';
            header.magic[1] = 'B';
            header.magic[2] = 'V';
            header.magic[3] = 'H';

            header.version[0] = 0;
            header.version[1] = 0;
            header.version[2] = 0;
            header.version[3] = 1;

            header.nodeNum = (uint32_t)m_threadedNodes[0].size();
            header.maxDepth = m_maxDepth;

            header.cntMtrlForVoxel = (uint32_t)mtrlMap.size();

            auto bbox = getBoundingbox();
            auto boxmin = bbox.minPos();
            auto boxmax = bbox.maxPos();

            header.boxmin[0] = boxmin.x;
            header.boxmin[1] = boxmin.y;
            header.boxmin[2] = boxmin.z;

            header.boxmax[0] = boxmax.x;
            header.boxmax[1] = boxmax.y;
            header.boxmax[2] = boxmax.z;
        }

        fwrite(&header, sizeof(header), 1, fp);

        // Write material information.
        {
            static const uint8_t zeros[4] = { 0 };

            for (auto it : mtrlMap) {
                int mtrlid = it.first;
                const auto& name = it.second;

                // id.
                fwrite(&mtrlid, sizeof(mtrlid), 1, fp);

                int len = (int)name.length();
                int alinedLen = ((len + 3) / 4) * 4;
                int fillZeroLen = alinedLen - len;

                // String size.
                fwrite(&alinedLen, sizeof(alinedLen), 1, fp);

                // String.
                const char* pstr = name.c_str();
                fwrite(pstr, 1, len, fp);

                // Fill zero.
                if (fillZeroLen > 0) {
                    fwrite(zeros, 1, fillZeroLen, fp);
                }
            }
        }

        // Nodes.
        fwrite(&m_threadedNodes[0][0], sizeof(ThreadedSbvhNode), header.nodeNum, fp);

        fclose(fp);

        return true;
    }

    bool sbvh::importTree(
        const context& ctxt,
        const char* path, 
        int offsetTriIdx)
    {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            // TODO
            // through exception...
            AT_ASSERT(false);
            return false;
        }

        SbvhFileHeader header;
        fread(&header, sizeof(header), 1, fp);

        // TODO
        // Check magic number.

        // Read material information.
        std::map<int, std::string> mtrlMap;
        {
            // TODO
            static char tmpbuf[128] = { 0 };

            for (int i = 0; i < header.cntMtrlForVoxel; i++)
            {
                // id.
                int mtrlid = -1;
                fread(&mtrlid, sizeof(mtrlid), 1, fp);

                // String size.
                int len = 0;
                fread(&len, sizeof(len), 1, fp);

                AT_ASSERT(0 < len && len < AT_COUNTOF(tmpbuf));

                // String.
                fread(tmpbuf, 1, len, fp);
                tmpbuf[len] = 0;    // Add termination.

                mtrlMap.insert(std::make_pair(mtrlid, std::string(tmpbuf)));
            }
        }

        m_threadedNodes.resize(1);
        m_threadedNodes[0].resize(header.nodeNum);

        fread(&m_threadedNodes[0][0], sizeof(ThreadedSbvhNode), header.nodeNum, fp);

        if (offsetTriIdx > 0 || !mtrlMap.empty())
        {
            for (auto& nodes : m_threadedNodes) {
                for (auto& node : nodes) {
                    if (node.triid >= 0) {
                        node.triid += offsetTriIdx;
                    }

                    // Re-set material id.
                    if (node.mtrlid >= 0) {
                        auto found = mtrlMap.find(node.mtrlid);
                        AT_ASSERT(found != mtrlMap.end());

                        // Find material index by name.
                        const auto& name = found->second;
                        int mtrlid = ctxt.findMaterialIdxByName(name.c_str());
                        AT_ASSERT(mtrlid >= 0);

                        // Replace current material index.
                        node.mtrlid = mtrlid;
                    }
                }
            }
        }

        vec3 boxmin = vec3(header.boxmin[0], header.boxmin[1], header.boxmin[2]);
        vec3 boxmax = vec3(header.boxmax[0], header.boxmax[1], header.boxmax[2]);
        setBoundingBox(aabb(boxmin, boxmax));

        m_maxDepth = header.maxDepth;

        m_isImported = true;

        return true;
    }

    static inline void _drawAABB(
        const aten::ThreadedSbvhNode* node,
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W)
    {
        aabb box(node->boxmin, node->boxmax);

        auto transofrmedBox = aten::aabb::transform(box, mtxL2W);

        aten::mat4 mtxScale;
        mtxScale.asScale(transofrmedBox.size());

        aten::mat4 mtxTrans;
        mtxTrans.asTrans(transofrmedBox.minPos());

        aten::mat4 mtx = mtxTrans * mtxScale;

        func(mtx);
    }

    void sbvh::drawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W)
    {
        // TODO
        //int nodeListSize = m_threadedNodes.size();
        int nodeListSize = 1;

        static const uint32_t stacksize = 64;
        const ThreadedSbvhNode* stackbuf[stacksize];

        for (int i = 0; i < nodeListSize; i++) {
            const auto& nodes = m_threadedNodes[i];

            auto* node = &nodes[0];

            stackbuf[0] = node;
            int stackpos = 1;

            while (stackpos > 0) {
                node = stackbuf[stackpos - 1];

                stackpos -= 1;

                _drawAABB(node, func, mtxL2W);

                int hit = (int)node->hit;
                int miss = (int)node->miss;

                if (hit >= 0) {
                    stackbuf[stackpos++] = &nodes[hit];
                }
                if (miss >= 0) {
                    stackbuf[stackpos++] = &nodes[miss];
                }
            }
        }
    }

    void sbvh::update(const context& ctxt)
    {
        m_bvh.update(ctxt);

        // Only for top layer...

        const auto& toplayer = m_bvh.getNodes()[0];

        AT_ASSERT(m_threadedNodes[0].size() == toplayer.size());
        
        memcpy(&m_threadedNodes[0][0], &toplayer[0], toplayer.size() * sizeof(ThreadedSbvhNode));
    }
}
