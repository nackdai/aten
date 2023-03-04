#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudaTextureResource.h"
#include "aten4idaten.h"

__forceinline__ __device__ float4 getFloat4(cudaTextureObject_t tex, int32_t idx)
{
    return tex1Dfetch<float4>(tex, idx);
}

__forceinline__ __device__ float4 getFloat4(float4* data, int32_t idx)
{
    return data[idx];
}

// NOTE
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
// http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/

// NOTE
// Extended Morton Codes(= EMC).
// http://dcgi.felk.cvut.cz/projects/emc/

__forceinline__ __device__ __host__ uint32_t expandBits(uint32_t value)
{
#if 0
    // NOTE
    // 0x00010001u = 1 << 16 + 1 = 0x00010000 + 1 = 0x00010001
    // 0x00000101u = 1 << 8 + 1 = 0x00000100 + 1 = 0x00000101
    // 0x00000011u = 1 << 4 + 1 = 0x00000010 + 1 = 0x00000011
    // 0x00000005u = 1 << 2 + 1 = 0x00000004 + 1 = 0x00000005
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
#else
    // NOTE
    // value | value << 16 = value * (1 + 1 << 16) = value * 0x00010001u
    value = (value | value << 16) & 0xFF0000FFu;
    value = (value | value << 8) & 0x0F00F00Fu;
    value = (value | value << 4) & 0xC30C30C3u;
    value = (value | value << 2) & 0x49249249u;
#endif
    return value;
}

__forceinline__ __device__ __host__ uint64_t expandBitsBy3(uint32_t value)
{
    uint64_t x = value;
    x = (x | x << 36) & 0x000f000000000ffful;
    x = (x | x << 24) & 0x000f000f000000fful;
    x = (x | x << 12) & 0x000f000f000f000ful;
    x = (x | x << 6) & 0x0303030303030303ul;
    x = (x | x << 3) & 0x1111111111111111ul;
    return x;
}

__forceinline__ __device__ __host__ uint32_t computeMortonCode(aten::vec3 point)
{
    // Discretize the unit cube into a 10 bit integer
    uint3 discretized;
    discretized.x = (uint32_t)min(max(point.x * 1024.0f, 0.0f), 1023.0f);
    discretized.y = (uint32_t)min(max(point.y * 1024.0f, 0.0f), 1023.0f);
    discretized.z = (uint32_t)min(max(point.z * 1024.0f, 0.0f), 1023.0f);

    discretized.x = expandBits(discretized.x);
    discretized.y = expandBits(discretized.y);
    discretized.z = expandBits(discretized.z);

#if 0
    return discretized.x * 4 + discretized.y * 2 + discretized.z;
#else
    return discretized.x << 2 | discretized.y << 1 | discretized.z;
#endif
}

__forceinline__ __device__ __host__ uint64_t computeMortonCode(float x, float y, float z, float s)
{
    // Discretize the unit cube into a 8 bit integer
    uint32_t dx = (uint32_t)min(max(x * 65536.0f, 0.0f), 65535.0f);
    uint32_t dy = (uint32_t)min(max(y * 65536.0f, 0.0f), 65535.0f);
    uint32_t dz = (uint32_t)min(max(z * 65536.0f, 0.0f), 65535.0f);
    uint32_t dw = (uint32_t)min(max(s * 65536.0f, 0.0f), 65535.0f);

    uint64_t ddx = expandBitsBy3(dx);
    uint64_t ddy = expandBitsBy3(dy);
    uint64_t ddz = expandBitsBy3(dz);
    uint64_t ddw = expandBitsBy3(dw);

    uint64_t ret = (uint64_t)(ddx << 3 | ddy << 2 | ddz << 1 | ddw);

    return ret;
}

template <typename T>
__forceinline__ __device__ T onComputeMortonCode(
    int32_t a0, int32_t a1, int32_t a2,
    const aten::vec3& vmin,
    const aten::vec3& vmax,
    const aten::aabb& sceneBbox)
{
    // Normalize [0, 1].
    const auto size = sceneBbox.size();
    const auto bboxMin = sceneBbox.minPos();

    aten::vec3 center = (vmin + vmax) * 0.5f;
    center = (center - bboxMin) / size;

    auto code = computeMortonCode(center);

    return code;
}

template <uint64_t>
__forceinline__ __device__ uint64_t onComputeExtendedMortonCode(
    int32_t a0, int32_t a1, int32_t a2,
    const aten::vec3& vmin,
    const aten::vec3& vmax,
    const aten::aabb& sceneBbox)
{
    // Normalize [0, 1].
    const auto size = sceneBbox.size();
    const auto bboxMin = sceneBbox.minPos();

    aten::vec3 center = (vmin + vmax) * 0.5f;
    center = (center - bboxMin) / size;

    // Compute diagonal.
    const auto div = sceneBbox.getDiagonalLenght();
    auto d = aten::length(vmax - vmin);
    d /= div;

    auto code = computeMortonCode(center[a0], center[a1], center[a2], d);

    return code;
}

template <typename T, typename M>
__global__ void genMortonCode(
    int32_t a0, int32_t a1, int32_t a2,
    int32_t numberOfTris,
    const aten::aabb sceneBbox,
    const aten::TriangleParameter* __restrict__ tris,
    T vtxPos,
    int32_t vtxOffset,
    M* mortonCodes,
    uint32_t* indices)
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= numberOfTris) {
        return;
    }

    aten::TriangleParameter prim;
    prim.v0 = ((aten::vec4*)tris)[idx * aten::TriangleParamter_float4_size + 0];

    float4 v0 = getFloat4(vtxPos, prim.idx[0] + vtxOffset);
    float4 v1 = getFloat4(vtxPos, prim.idx[1] + vtxOffset);
    float4 v2 = getFloat4(vtxPos, prim.idx[2] + vtxOffset);

    aten::vec3 vmin = aten::vec3(
        min(min(v0.x, v1.x), v2.x),
        min(min(v0.y, v1.y), v2.y),
        min(min(v0.z, v1.z), v2.z));

    aten::vec3 vmax = aten::vec3(
        max(max(v0.x, v1.x), v2.x),
        max(max(v0.y, v1.y), v2.y),
        max(max(v0.z, v1.z), v2.z));

    aten::vec3 center = (vmin + vmax) * 0.5f;

    // Normalize [0, 1].
    const auto size = sceneBbox.size();
    const auto bboxMin = sceneBbox.minPos();
    center = (center - bboxMin) / size;

    auto code = onComputeMortonCode<M>(
        a0, a1, a2,
        vmin, vmax, sceneBbox);

    mortonCodes[idx] = code;
    indices[idx] = idx;
}
