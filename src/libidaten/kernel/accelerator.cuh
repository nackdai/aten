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
#elif defined(GPGPU_TRAVERSE_STACKLESS_QBVH)
	#include "kernel/stackless_qbvh.cuh"
	#define intersectClosest	intersectClosestStacklessQBVH
	#define intersectCloser		intersectCloserStacklessQBVH
	#define interserctAny		intersectAnyStacklessQBVH
#elif defined(GPGPU_TRAVERSE_SBVH)
	#include "kernel/sbvh.cuh"
	#define intersectClosest	intersectClosestSBVH
	#define intersectCloser		intersectCloserSBVH
	#define interserctAny		intersectAnySBVH
#endif