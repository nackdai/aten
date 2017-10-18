#pragma once

#include "kernel/bvh.cuh"
#include "kernel/qbvh.cuh"

#if defined(GPGPU_TRAVERSE_THREADED_BVH)
	#define intersectClosest	intersectBVH
	#define intersectCloser		intersectCloserBVH
	#define interserctAny		intersectAnyBVH
#elif defined(GPGPU_TRAVERSE_QBVH)
	#define intersectClosest	intersectQBVH
	#define intersectCloser		intersectCloserQBVH
	#define interserctAny		intersectAnyQBVH
#endif