#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudaTextureResource.h"

#include "aten4idaten.h"

__forceinline__ __device__ float4 getFloat4(cudaTextureObject_t tex, int idx)
{
	return tex1Dfetch<float4>(tex, idx);
}

__forceinline__ __device__ float4 getFloat4(float4* data, int idx)
{
	return data[idx];
}

// NOTE
// http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

__forceinline__ __device__ __host__ unsigned int expandBits(unsigned int value)
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

__forceinline__ __device__ __host__ unsigned int computeMortonCode(aten::vec3 point)
{
	// Discretize the unit cube into a 10 bit integer
	uint3 discretized;
	discretized.x = (unsigned int)min(max(point.x * 1024.0f, 0.0f), 1023.0f);
	discretized.y = (unsigned int)min(max(point.y * 1024.0f, 0.0f), 1023.0f);
	discretized.z = (unsigned int)min(max(point.z * 1024.0f, 0.0f), 1023.0f);

	discretized.x = expandBits(discretized.x);
	discretized.y = expandBits(discretized.y);
	discretized.z = expandBits(discretized.z);

#if 0
	return discretized.x * 4 + discretized.y * 2 + discretized.z;
#else
	return discretized.x << 2 | discretized.y << 1 | discretized.z;
#endif
}

template <typename T>
__global__ void genMortonCode(
	int numberOfTris,
	const aten::aabb sceneBbox,
	const aten::PrimitiveParamter* __restrict__ tris,
	T vtxPos,
	int vtxOffset,
	uint32_t* mortonCodes,
	uint32_t* indices)
{
	const auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numberOfTris) {
		return;
	}

	aten::PrimitiveParamter prim;
	prim.v0 = ((aten::vec4*)tris)[idx * aten::PrimitiveParamter_float4_size + 0];

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

	auto code = computeMortonCode(center);

	mortonCodes[idx] = code;
	indices[idx] = idx;
}
