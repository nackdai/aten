#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

struct BVHRay : public aten::ray {
	aten::vec3 inv;
	int sign[3];

	__device__ BVHRay(const aten::ray& r)
	{
		org = r.org;
		dir = r.dir;

		inv = real(1) / dir;

		sign[0] = (inv.x < real(0) ? 1 : 0);
		sign[1] = (inv.y < real(0) ? 1 : 0);
		sign[2] = (inv.z < real(0) ? 1 : 0);
	}
};

enum IntersectType {
	Closest,
	Closer,
	Any,
};

__device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect);

__device__ bool intersectCloserBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max);

#ifndef __AT_DEBUG__
#include "kernel/bvh_impl.cuh"
#endif
