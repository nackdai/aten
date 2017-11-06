#pragma once

#include "accelerator/threaded_bvh.h"
#include "accelerator/qbvh.h"
#include "accelerator/stackless_bvh.h"

#define GPGPU_TRAVERSE_THREADED_BVH
//#define GPGPU_TRAVERSE_QBVH
//#define GPGPU_TRAVERSE_STACKLESS_BVH

namespace aten {
#if defined(GPGPU_TRAVERSE_THREADED_BVH)
	using GPUBvhNode = ThreadedBvhNode;
	using GPUBvh = ThreadedBVH;
#elif defined(GPGPU_TRAVERSE_QBVH)
	using GPUBvhNode = QbvhNode;
	using GPUBvh = qbvh;
#elif defined(GPGPU_TRAVERSE_STACKLESS_BVH)
	using GPUBvhNode = StacklessBvhNode;
	using GPUBvh = StacklessBVH;
#else
	AT_STATICASSERT(false);
#endif

	AT_STATICASSERT((sizeof(GPUBvhNode) % (sizeof(float) * 4)) == 0);
	static const int GPUBvhNodeSize = sizeof(GPUBvhNode) / (sizeof(float) * 4);
}