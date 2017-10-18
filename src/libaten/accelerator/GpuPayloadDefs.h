#pragma once

#include "accelerator/threaded_bvh.h"
#include "accelerator/qbvh.h"

// TODO
#define GPGPU_TRAVERSE_THREADED_BVH
#define GPGPU_TRAVERSE_QBVH

namespace aten {
#if defined(GPGPU_TRAVERSE_THREADED_BVH)
	using GPUBvhNode = ThreadedBvhNode;
#elif defined(GPGPU_TRAVERSE_QBVH)
	using GPUBvhNode = QbvhNode;
#else
	AT_STATICASSERT(false);
#endif
}