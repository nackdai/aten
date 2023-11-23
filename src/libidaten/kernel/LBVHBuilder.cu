#include "kernel/LBVHBuilder.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudaTextureResource.h"
#include "accelerator/GpuPayloadDefs.h"
#include "kernel/MortonCode.cuh"

//#pragma optimize( "", off)

// NOTE
// https://github.com/leonardo-domingues/atrbvh
// http://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf

#if 0
template <class T>
__forceinline__ __device__ int32_t computeLongestCommonPrefix(
    const T* sortedKeys,
    uint32_t numOfElems,
    int32_t index1, int32_t index2,
    T key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one
    // thread per internal node)
    if (index2 < 0 || index2 >= numOfElems)
    {
        // NOTE
        // delta(i, j) = ?1 when not (0 <= j <= n - 1).
        return -1;
    }

    auto key2 = sortedKeys[index2];

    if (key1 == key2)
    {
        return 32 + __clz(index1 ^ index2);
    }

    auto ret = __clz(key1 ^ key2);

    return ret;
}

template <>
__forceinline__ __device__ int32_t computeLongestCommonPrefix(
    const uint64_t* sortedKeys,
    uint32_t numOfElems,
    int32_t index1, int32_t index2,
    uint64_t key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one
    // thread per internal node)
    if (index2 < 0 || index2 >= numOfElems)
    {
        // NOTE
        // delta(i, j) = ?1 when not (0 <= j <= n - 1).
        return -1;
    }

    auto key2 = sortedKeys[index2];

    if (key1 == key2)
    {
        return 63 + __clz(index1 ^ index2);
    }

    auto ret = __clzll(key1 ^ key2);

    return ret;
}

template <class T>
__global__ void buildTree(
    uint32_t numOfElems,
    const T* __restrict__ sortedKeys,
    idaten::LBVHBuilder::LBVHNode* nodes)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numOfElems - 1) {
        return;
    }

    const auto key1 = sortedKeys[i];

    const auto lcp1 = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + 1, key1);
    const auto lcp2 = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i - 1, key1);

    // どちら向きに探索していくかを決める.
    // CommonPrefix が長くなる方向に探索する.
    auto d = (lcp1 - lcp2) < 0 ? -1 : 1;

    // Compute upper bound for the length of the range
    // 探索範囲の上限を決める. 倍々に広げていき、下限基準より LongestCommonPrefix が長くなる位置を探索範囲の上限とする.

    // Common Prefix が長くなる方向とは１つ反対の LogestCommonPrefix を計算する.
    // 長くなる方向とは１つ反対 = LogestCommonPrefix の下限基準.
    const auto minLcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i - d, key1);
    int32_t lMax = 2;
    while (computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + lMax * d, key1) > minLcp)
    {
        lMax *= 2;
    }

    // Find other end using binary search
    // 2分探索で厳密に上限を決める.
    int32_t lowest = 0;
    int32_t t = lMax;
    while (t > 1)
    {
        t = t / 2;

        auto lcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + (lowest + t) * d, key1);
        if (lcp > minLcp)
        {
            // より長いLogestCommonPrefixが見つかったので、位置をそこに移動.
            lowest += t;
        }
    }

    // 探索範囲の上限.
    const auto j = i + lowest * d;

    // Find the split position using binary search
    // 分割位置を2分探索で決める.

    // 探索範囲の下限と上限の間のLongestCommonPrefixを計算.
    const auto nodeLcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, j, key1);

    int32_t start = 0;
    int32_t divisor = 2;
    t = lowest;

    while (t > 1)
    {
        t = (lowest + divisor - 1) / divisor;

        auto lcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + (start + t) * d, key1);
        if (lcp > nodeLcp)
        {
            // より長いLogestCommonPrefixが見つかったので、位置をそこに移動.
            start += t;
        }
        divisor *= 2;
    }

    const auto splitPosition = i + start * d + min(d, 0);

    auto* node = nodes + i;
    if (i == 0) {
        node->parent = -1;
    }
    node->order = i;
    node->isLeaf = false;

    // End of internal nodes position = Start of leaf node position.
    uint32_t leafBaseIdx = numOfElems - 1;

    if (min(i, j) == splitPosition) {
        node->left = leafBaseIdx + splitPosition;

        auto* leaf = nodes + node->left;
        leaf->order = node->left;
        leaf->parent = i;
        leaf->left = -1;
        leaf->right = -1;
        leaf->isLeaf = true;
    }
    else {
        node->left = splitPosition;

        auto* child = nodes + node->left;
        child->order = node->left;
        child->parent = i;
        child->isLeaf = false;
    }

    if (max(i, j) == splitPosition + 1) {
        node->right = leafBaseIdx + splitPosition + 1;

        auto* leaf = nodes + node->right;
        leaf->order = node->right;
        leaf->parent = i;
        leaf->left = -1;
        leaf->right = -1;
        leaf->isLeaf = true;
    }
    else {
        node->right = splitPosition + 1;

        auto* child = nodes + node->right;
        child->order = node->right;
        child->parent = i;
        child->isLeaf = false;
    }
}
#else
__forceinline__ __device__ int32_t computeLongestCommonPrefix(
    const uint32_t* sortedKeys,
    uint32_t numOfElems,
    int32_t index1, int32_t index2)
{
    // Select left end
    int32_t left = min(index1, index2);

    // Select right end
    int32_t right = max(index1, index2);

    // This is to ensure the node breaks if the index is out of bounds
    if (left < 0 || right >= numOfElems)
    {
        return -1;
    }
    // Fetch Morton codes for both ends
    int32_t left_code = sortedKeys[left];
    int32_t right_code = sortedKeys[right];

    // Special handling of duplicated codes: use their indices as a fallback
    return left_code != right_code ? __clz(left_code ^ right_code) : (32 + __clz(left ^ right));
}

__forceinline__ __device__ int3 findSpan(
    const uint32_t* mortonCodes,
    uint32_t numPrims,
    int32_t idx)
{
    auto lcp1 = computeLongestCommonPrefix(mortonCodes, numPrims, idx, idx + 1);
    auto lcp2 = computeLongestCommonPrefix(mortonCodes, numPrims, idx, idx - 1);

    // どちら向きに探索していくかを決める.
    // CommonPrefix が長くなる方向に探索する.
    int32_t d = (lcp1 - lcp2) < 0 ? -1 : 1;

    // 探索範囲の上限を決める. 倍々に広げていき、下限基準より LongestCommonPrefix が長くなる位置を探索範囲の上限とする.

    // Find minimum number of bits for the break on the other side.
    // Common Prefix が長くなる方向とは１つ反対の LogestCommonPrefix を計算する.
    // 長くなる方向とは１つ反対 = LogestCommonPrefix の下限基準.
    int32_t minLcp = computeLongestCommonPrefix(mortonCodes, numPrims, idx, idx - d);

    // Search conservative far end
    int32_t lmax = 2;
    while (computeLongestCommonPrefix(mortonCodes, numPrims, idx, idx + lmax * d) > minLcp) {
        lmax *= 2;
    }

    // Search back to find exact bound with binary search.
    // 2分探索で厳密に上限を決める.
    int32_t l = 0;
    int32_t t = lmax;
    do
    {
        t /= 2;
        if (computeLongestCommonPrefix(mortonCodes, numPrims, idx, idx + (l + t) * d) > minLcp)
        {
            l = l + t;
        }
    } while (t > 1);

    // Pack span
    int3 span;
    span.x = min(idx, idx + l * d);
    span.y = max(idx, idx + l * d);
    span.z = d;
    return span;
}

// Find split idx within the span
__forceinline__ __device__ int32_t findSplit(
    const uint32_t* sortedKeys,
    uint32_t numOfElems,
    int3 span)
{
    // Fetch codes for both ends
    int32_t left = span.x;
    int32_t right = span.y;
    int32_t d = span.z;

    // Calculate the number of identical bits from higher end
    int32_t numIdentical = computeLongestCommonPrefix(sortedKeys, numOfElems, left, right);

    do
    {
        // Proposed split
        int32_t newSplit = (right + left) / 2;

        // If it has more equal leading bits than left and right accept it
        if (computeLongestCommonPrefix(sortedKeys, numOfElems, left, newSplit) > numIdentical)
        {
            left = newSplit;
        }
        else
        {
            right = newSplit;
        }
    } while (right > left + 1);

    return left;
}

__global__ void buildTree(
    uint32_t numOfElems,
    const uint32_t* __restrict__ sortedKeys,
    idaten::LBVHBuilder::LBVHNode* nodes)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numOfElems - 1) {
        return;
    }

    auto range = findSpan(sortedKeys, numOfElems, idx);

    auto split = findSplit(sortedKeys, numOfElems, range);

    auto* node = nodes + idx;
    if (idx == 0) {
        node->parent = -1;
    }
    node->order = idx;
    node->isLeaf = false;

    // Create child nodes if needed
    if (split == range.x) {
        node->left = split + numOfElems - 1;

        auto* leaf = nodes + node->left;
        leaf->order = node->left;
        leaf->parent = idx;
        leaf->left = -1;
        leaf->right = -1;
        leaf->isLeaf = true;
    }
    else {
        node->left = split;

        auto* child = nodes + node->left;
        child->order = node->left;
        child->parent = idx;
        child->isLeaf = false;
    }

    if (split + 1 == range.y) {
        node->right = split + 1 + numOfElems - 1;

        auto* leaf = nodes + node->right;
        leaf->order = node->right;
        leaf->parent = idx;
        leaf->left = -1;
        leaf->right = -1;
        leaf->isLeaf = true;
    }
    else {
        node->right = split + 1;

        auto* child = nodes + node->right;
        child->order = node->right;
        child->parent = idx;
        child->isLeaf = false;
    }
}
#endif

__device__ __host__ inline void onApplyTraverseOrder(
    int32_t idx,
    int32_t numberOfTris,
    int32_t triIdOffset,
    const idaten::LBVHBuilder::LBVHNode* __restrict__ src,
    const uint32_t* __restrict__ sortedIndices,
    aten::ThreadedBvhNode* dst)
{
    const auto* node = &src[idx];

    const idaten::LBVHBuilder::LBVHNode* next = node->left >= 0 ? &src[node->left] : nullptr;

    auto gpunode = &dst[idx];

    gpunode->object_id = -1;
    gpunode->exid = -1;
    gpunode->meshid = -1;

    if (node->isLeaf) {
        // Base index to convert node index to triangle index.
        int32_t leafBaseIdx = numberOfTris - 1;

        int32_t leafId = node->order - leafBaseIdx;
        int32_t triId = triIdOffset + sortedIndices[leafId];

        gpunode->primid = (float)triId;

#if defined(GPGPU_TRAVERSE_SBVH)
        // For ThreadedSbvhNode, this is "isleaf".
        gpunode->object_id = 1;
#endif
    }
    else {
        gpunode->primid = -1;
    }

    gpunode->hit = -1;
    gpunode->miss = -1;

    bool isOrdered = false;

    if (node->isLeaf) {
        // Hit/Miss.
        // Always sibling.

        // The leaf has parent surely.
        auto parent = &src[node->parent];

        auto left = parent->left >= 0 ? &src[parent->left] : nullptr;
        auto right = parent->right >= 0 ? &src[parent->right] : nullptr;

        if (left == node) {
            // Sibling.
            gpunode->hit = (float)right->order;
            gpunode->miss = (float)right->order;

            isOrdered = true;
        }
    }
    else {
        // Hit.
        // Always the next node in the array.
        if (next) {
            gpunode->hit = (float)next->order;
        }
        else {
            gpunode->hit = -1;
        }
    }

    if (!isOrdered)
    {
        // Miss.

        // Search the parent.
        auto parentId = node->parent;
        const auto parent = (parentId >= 0
            ? &src[parentId]
            : nullptr);

        if (parent) {
            const auto left = parent->left >= 0 ? &src[parent->left] : nullptr;
            const auto right = parent->right >= 0 ? &src[parent->right] : nullptr;

            if (left == node && right) {
                // Traverse to sibling (= parent's right)
                auto sibling = right;
                gpunode->miss = (float)sibling->order;
            }
            else {
                auto curParent = parent;

                // Traverse to ancester's right.
                for (;;) {
                    // Search the grand parent.
                    auto grandParentId = curParent->parent;
                    const auto grandParent = (grandParentId >= 0
                        ? &src[grandParentId]
                        : nullptr);

                    if (grandParent) {
                        const auto _left = grandParent->left >= 0 ? &src[grandParent->left] : nullptr;
                        const auto _right = grandParent->right >= 0 ? &src[grandParent->right] : nullptr;

                        auto sibling = _right;

                        if (sibling) {
                            if (sibling != curParent) {
                                gpunode->miss = (float)sibling->order;

                                if (node->isLeaf && gpunode->hit < 0) {
                                    gpunode->hit = (float)sibling->order;
                                }

                                break;
                            }
                        }
                    }
                    else {
                        gpunode->miss = -1;
                        break;
                    }

                    curParent = grandParent;
                }
            }
        }
        else {
            gpunode->miss = -1;
        }
    }
}

__global__ void applyTraverseOrder(
    uint32_t numberOfNodes,
    int32_t numberOfTris,
    int32_t triIdOffset,
    const idaten::LBVHBuilder::LBVHNode* __restrict__ src,
    const uint32_t* __restrict__ sortedIndices,
    aten::ThreadedBvhNode* dst)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numberOfNodes) {
        return;
    }

    onApplyTraverseOrder(idx, numberOfTris, triIdOffset, src, sortedIndices, dst);
}

__device__ inline void computeBoundingBox(
    float4 v0,
    float4 v1,
    float4 v2,
    float4* aabbMin,
    float4* aabbMax)
{
    aabbMin->x = min(min(v0.x, v1.x), v2.x);
    aabbMin->y = min(min(v0.y, v1.y), v2.y);
    aabbMin->z = min(min(v0.z, v1.z), v2.z);

    aabbMax->x = max(max(v0.x, v1.x), v2.x);
    aabbMax->y = max(max(v0.y, v1.y), v2.y);
    aabbMax->z = max(max(v0.z, v1.z), v2.z);
}

__device__ inline void computeBoundingBox(
    float4 bboxMin_0,
    float4 bboxMax_0,
    float4 bboxMin_1,
    float4 bboxMax_1,
    float4* aabbMin,
    float4* aabbMax)
{
    aabbMin->x = min(bboxMin_0.x, bboxMin_1.x);
    aabbMin->y = min(bboxMin_0.y, bboxMin_1.y);
    aabbMin->z = min(bboxMin_0.z, bboxMin_1.z);

    aabbMax->x = max(bboxMax_0.x, bboxMax_1.x);
    aabbMax->y = max(bboxMax_0.y, bboxMax_1.y);
    aabbMax->z = max(bboxMax_0.z, bboxMax_1.z);
}

template <class T>
__global__ void computeBoudingBox(
    int32_t numberOfTris,
    const idaten::LBVHBuilder::LBVHNode* __restrict__ src,
    const uint32_t* __restrict__ sortedIndices,
    aten::ThreadedBvhNode* dst,
    const aten::TriangleParameter* __restrict__ tris,
    T vtxPos,
    int32_t vtxOffset,
    uint32_t* executedIdxArray)
{
    const int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int32_t firstThreadIdxInBlock = blockIdx.x * blockDim.x;
    const int32_t lastThreadIdxInBlock = firstThreadIdxInBlock + blockDim.x - 1;

    // Initialize cache of bounding boxes in shared memory
    extern __shared__ float4 sharedBboxMin[];
    __shared__ float4* sharedBboxMax;
    if (threadIdx.x == 0)
    {
        sharedBboxMax = sharedBboxMin + blockDim.x;
    }
    __syncthreads();

    // Check for valid threads
    if (idx >= numberOfTris)
    {
        return;
    }

    // NOTE
    // Number of Internal Nodes = Number of Triangles - 1.
    int32_t leafNodeIdx = idx + numberOfTris - 1;

    // Base index to convert node index to triangle index.
    int32_t leafBaseIdx = numberOfTris - 1;

    const auto* node = &src[leafNodeIdx];
    auto* gpunode = &dst[leafNodeIdx];

    // Calculate leaves bounding box.
    int32_t leafId = node->order - leafBaseIdx;
    int32_t triId = sortedIndices[leafId];

    aten::TriangleParameter prim;
    prim.v0 = ((aten::vec4*)tris)[triId * aten::TriangleParamter_float4_size + 0];

    float4 v0 = getFloat4(vtxPos, prim.idx[0] + vtxOffset);
    float4 v1 = getFloat4(vtxPos, prim.idx[1] + vtxOffset);
    float4 v2 = getFloat4(vtxPos, prim.idx[2] + vtxOffset);

    float4 aabbMin, aabbMax;
    computeBoundingBox(v0, v1, v2, &aabbMin, &aabbMax);

    // Keep bouding box to shared memory.
    sharedBboxMin[threadIdx.x] = aabbMin;
    sharedBboxMax[threadIdx.x] = aabbMax;

    gpunode->boxmin = aten::vec3(aabbMin.x, aabbMin.y, aabbMin.z);
    gpunode->boxmax = aten::vec3(aabbMax.x, aabbMax.y, aabbMax.z);

#if 0
    printf("Vtx(%d : %d %d %d) [%f, %f, %f] [%f, %f, %f] [%f, %f, %f]\n",
        triId,
        prim.idx[0], prim.idx[1], prim.idx[2],
        v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
    printf("Target[%d] [%f, %f, %f] [%f, %f, %f]\n", leafNodeIdx, aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z);
#endif

    __syncthreads();

    // リーフから親へたどっていく.

    int32_t lastNode = idx;
    int32_t targetId = node->parent;

    while (targetId >= 0)
    {
        // ターゲットは親ノードで、ここでは子ノードを処理しているであろうスレッドのインデックスを取得する.
        // インデックスの配列は 0xffffffff で初期化されていて、処理されたらスレッドのインデックスで置換される.
        // つまり、配列内の値が 0xffffffff であったら、未処理ということになる.
        const auto childNodeThreadIdx = atomicExch(&executedIdxArray[targetId], idx);

        if (childNodeThreadIdx == 0xffffffff) {
            // 未処理なので、これ以上は何もしない.
            return;
        }

        const auto* targetSrc = &src[targetId];
        auto* targetDst = &dst[targetId];

        float4 childAABBMin, childAABBMax;

        // 子ノードを処理しているであろうスレッドが同一ブロックで処理しているかどうか.
        if (firstThreadIdxInBlock <= childNodeThreadIdx
            && childNodeThreadIdx <= lastThreadIdxInBlock)
        {
            // 同一ブロックで処理されているので、shared memory にキャッシュされているデータを取得する.

            // ブロック内でのスレッドIDに変換.
            int32_t threadIdxInBlock = childNodeThreadIdx - firstThreadIdxInBlock;

            childAABBMin = sharedBboxMin[threadIdxInBlock];
            childAABBMax = sharedBboxMax[threadIdxInBlock];
        }
        else {
            // 同一ブロックで処理されていないので、配列に格納されているデータを取得する.

            int32_t childIdx = targetSrc->left;

            if (childIdx == lastNode) {
                childIdx = targetSrc->right;
            }

            const auto* tmp = &dst[childIdx];

            childAABBMin = make_float4(tmp->boxmin.x, tmp->boxmin.y, tmp->boxmin.z, 0);
            childAABBMax = make_float4(tmp->boxmax.x, tmp->boxmax.y, tmp->boxmax.z, 0);
        }

        __syncthreads();

        computeBoundingBox(
            aabbMin, aabbMax,
            childAABBMin, childAABBMax,
            &aabbMin, &aabbMax);

        // Keep bouding box to shared memory.
        sharedBboxMin[threadIdx.x] = aabbMin;
        sharedBboxMax[threadIdx.x] = aabbMax;

        targetDst->boxmin = aten::vec3(aabbMin.x, aabbMin.y, aabbMin.z);
        targetDst->boxmax = aten::vec3(aabbMax.x, aabbMax.y, aabbMax.z);

        //printf("Target[%d] [%f, %f, %f] [%f, %f, %f]\n", targetId, aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z);

        __syncthreads();

        // Update last processed node
        lastNode = targetId;

        // Update target node pointer
        targetId = targetSrc->parent;
    }
}

namespace idaten
{
    void LBVHBuilder::init(uint32_t maxNum)
    {
        m_mortonCodes.resize(maxNum);
        m_indices.resize(maxNum);

#ifdef AT_ENABLE_64BIT_LBVH_MORTON_CODE
        m_sort.initWith64Bit(maxNum);
#else
        m_sort.init(maxNum);
#endif

        uint32_t numInternalNode = maxNum - 1;
        uint32_t numLeaves = maxNum;

        m_nodesLbvh.resize(numInternalNode + numLeaves);

        m_nodes.resize(numInternalNode + numLeaves);

        if (!m_executedIdxArray) {
            checkCudaErrors(cudaMalloc(&m_executedIdxArray, (maxNum - 1) * sizeof(uint32_t)));
        }
    }

    template <class T>
    void LBVHBuilder::onBuild(
        idaten::CudaTextureResource& dst,
        TypedCudaMemory<aten::TriangleParameter>& triangles,
        int32_t triIdOffset,
        const aten::aabb& sceneBbox,
        T vtxPos,
        int32_t vtxOffset,
        std::vector<aten::ThreadedBvhNode>* threadedBvhNodes)
    {
        uint32_t numberOfTris = (uint32_t)triangles.num();

        // Get longest axis order.
        int32_t axis[3] = { 0, 1, 2 };
        {
            const auto size = sceneBbox.size();
            if (size[axis[0]] < size[axis[1]]) {
                std::swap(axis[0], axis[1]);
            }
            if (size[axis[1]] < size[axis[2]]) {
                std::swap(axis[1], axis[2]);
            }
            if (size[axis[0]] < size[axis[1]]) {
                std::swap(axis[0], axis[1]);
            }
        }

        // Compute morton code.
        {
            dim3 block(256, 1, 1);
            dim3 grid((numberOfTris + block.x - 1) / block.x, 1, 1);

            genMortonCode << <grid, block >> > (
                axis[0], axis[1], axis[2],
                numberOfTris,
                sceneBbox,
                triangles.data(),
                vtxPos,
                vtxOffset,
                m_mortonCodes.data(),
                m_indices.data());

            checkCudaKernel(genMortonCode);
        }

        // Radix sort.
#ifdef AT_ENABLE_64BIT_LBVH_MORTON_CODE
        m_sort.sortWith64Bit(
#else
        m_sort.sort(
#endif
            numberOfTris,
            m_mortonCodes,
            m_indices);

        uint32_t numInternalNode = numberOfTris - 1;
        uint32_t numLeaves = numberOfTris;

        // Build tree.
        {
            dim3 block(256, 1, 1);
            dim3 grid((numberOfTris + block.x - 1) / block.x, 1, 1);

            buildTree << <grid, block >> > (
                numberOfTris,
                m_mortonCodes.data(),
                m_nodesLbvh.data());

            checkCudaKernel(buildTree);
        }

        // Convert to gpu bvh tree nodes.
        {
            const auto numOfElems = numInternalNode + numLeaves;

            dim3 block(128, 1, 1);
            dim3 grid((numOfElems + block.x - 1) / block.x, 1, 1);

            applyTraverseOrder << <grid, block >> > (
                numOfElems,
                numLeaves,
                triIdOffset,
                m_nodesLbvh.data(),
                m_indices.data(),
                m_nodes.data());

            checkCudaKernel(applyTraverseOrder);
        }

        // Compute bouding box.
        {
            dim3 block(128, 1, 1);
            dim3 grid((numberOfTris + block.x - 1) / block.x, 1, 1);

            size_t sharedMemorySize = block.x * sizeof(float4) * 2;

            // Fill 0xff.
            checkCudaErrors(cudaMemset(m_executedIdxArray, 0xFF, (numberOfTris - 1) * sizeof(uint32_t)));

            computeBoudingBox << <grid, block, sharedMemorySize >> > (
                numberOfTris,
                m_nodesLbvh.data(),
                m_indices.data(),
                m_nodes.data(),
                triangles.data(),
                vtxPos,
                vtxOffset,
                m_executedIdxArray);

            checkCudaKernel(computeBoudingBox);
        }

        if (threadedBvhNodes) {
            auto num = numInternalNode + numLeaves;
            threadedBvhNodes->clear();
            threadedBvhNodes->resize(num);
            m_nodes.readFromDeviceToHostByNum(&(*threadedBvhNodes)[0], num);
        }

        dst.init(
            (aten::vec4*)m_nodes.data(),
            sizeof(aten::ThreadedBvhNode) / sizeof(float4),
            numInternalNode + numLeaves);
    }

    void LBVHBuilder::build(
        idaten::CudaTextureResource& dst,
        std::vector<aten::TriangleParameter>& tris,
        int32_t triIdOffset,
        const aten::aabb& sceneBbox,
        idaten::CudaTextureResource& texRscVtxPos,
        int32_t vtxOffset,
        std::vector<aten::ThreadedBvhNode>* threadedBvhNodes/*= nullptr*/)
    {
        TypedCudaMemory<aten::TriangleParameter> triangles;

        uint32_t numOfElems = (uint32_t)tris.size();

        triangles.resize(numOfElems);
        triangles.writeFromHostToDeviceByNum(&tris[0], (uint32_t)tris.size());

        auto vtxPos = texRscVtxPos.bind();

        onBuild(dst, triangles, triIdOffset, sceneBbox, vtxPos, vtxOffset, threadedBvhNodes);

        texRscVtxPos.unbind();
    }

    void LBVHBuilder::build(
        idaten::CudaTextureResource& dst,
        TypedCudaMemory<aten::TriangleParameter>& triangles,
        int32_t triIdOffset,
        const aten::aabb& sceneBbox,
        CudaGLBuffer& vboVtxPos,
        int32_t vtxOffset,
        std::vector<aten::ThreadedBvhNode>* threadedBvhNodes/*= nullptr*/)
    {
        vboVtxPos.map();

        float4* vtxPos = nullptr;
        size_t bytes = 0;
        vboVtxPos.bind((void**)&vtxPos, bytes);

        onBuild(dst, triangles, triIdOffset, sceneBbox, vtxPos, vtxOffset, threadedBvhNodes);

        vboVtxPos.unbind();
        vboVtxPos.unmap();
    }

    void LBVHBuilder::build()
    {
        static const uint32_t skeys[] = {
            1, 19, 24, 25, 30, 2, 4, 5,
        };

        std::vector<uint32_t> keys;
        std::vector<uint32_t> values;
        for (int32_t i = 0; i < AT_COUNTOF(skeys); i++) {
            keys.push_back(skeys[i]);
            values.push_back(i);
        }

        TypedCudaMemory<uint32_t> sortedKeys;
        TypedCudaMemory<uint32_t> sortedValue;
        std::vector<uint32_t> k;
        std::vector<uint32_t> v;
        RadixSort::sort(keys, values, sortedKeys, sortedValue, &k, &v);

        uint32_t numOfElems = values.size();

        uint32_t numInternalNode = numOfElems - 1;
        uint32_t numLeaves = numOfElems;

        TypedCudaMemory<LBVHNode> nodesLbvh;
        nodesLbvh.resize(numInternalNode + numLeaves);

        {
            dim3 block(256, 1, 1);
            dim3 grid((numOfElems + block.x - 1) / block.x, 1, 1);

            buildTree << <grid, block >> > (
                numOfElems,
                sortedKeys.data(),
                nodesLbvh.data());
        }

        std::vector<LBVHNode> tmp0(nodesLbvh.num());
        nodesLbvh.readFromDeviceToHostByNum(&tmp0[0]);

        TypedCudaMemory<aten::ThreadedBvhNode> nodes;
        nodes.resize(numInternalNode + numLeaves);

#if 1
        {
            numOfElems = numInternalNode + numLeaves;

            dim3 block(256, 1, 1);
            dim3 grid((numOfElems + block.x - 1) / block.x, 1, 1);

            applyTraverseOrder << <grid, block >> > (
                numOfElems,
                numLeaves,
                0,
                nodesLbvh.data(),
                sortedValue.data(),
                nodes.data());
        }
        std::vector<aten::ThreadedBvhNode> tmp1(nodes.num());
        nodes.readFromDeviceToHostByNum(&tmp1[0], 0);
#else
        std::vector<LBVHNode> tmp(m_nodesLbvh.maxNum());
        m_nodesLbvh.read(&tmp[0], 0);

        std::vector<aten::ThreadedBvhNode> tmp1(m_nodes.maxNum());

        for (int32_t n = 0; n < numInternalNode + numLeaves; n++) {
            onApplyTraverseOrder(n, &tmp[0], &tmp1[0]);
        }
#endif
    }
}
