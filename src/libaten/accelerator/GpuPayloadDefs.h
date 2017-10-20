#pragma once

#include "accelerator/threaded_bvh.h"
#include "accelerator/qbvh.h"

#define GPGPU_TRAVERSE_THREADED_BVH
//#define GPGPU_TRAVERSE_QBVH

namespace aten {
#if defined(GPGPU_TRAVERSE_THREADED_BVH)
	using GPUBvhNode = ThreadedBvhNode;
#elif defined(GPGPU_TRAVERSE_QBVH)
	using GPUBvhNode = QbvhNode;
#else
	AT_STATICASSERT(false);
#endif

	AT_STATICASSERT((sizeof(GPUBvhNode) % (sizeof(float) * 4)) == 0);
	static const int GPUBvhNodeSize = sizeof(GPUBvhNode) / (sizeof(float) * 4);
}