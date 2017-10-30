#pragma once

#if defined(GPGPU_TRAVERSE_THREADED_BVH)
	#include "kernel/bvh.cuh"
	#define intersectClosest	intersectBVH
	#define intersectCloser		intersectCloserBVH
	#define interserctAny		intersectAnyBVH
#elif defined(GPGPU_TRAVERSE_QBVH)
	#include "kernel/qbvh.cuh"
	#define intersectClosest	intersectQBVH
	#define intersectCloser		intersectCloserQBVH
	#define interserctAny		intersectAnyQBVH
#elif defined(GPGPU_TRAVERSE_STACKLESS_BVH)
	#include "kernel/stackless_bvh.cuh"
	#define intersectClosest	intersectClosestStacklessBVH
	#define intersectCloser		intersectCloserStacklessBVH
	#define interserctAny		intersectAnyStacklessBVH
#endif