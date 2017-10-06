#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

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

__device__ bool intersectAnyBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect);

#ifndef __AT_DEBUG__
#include "kernel/bvh_impl.cuh"
#endif
