#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#include "kernel/intersecttype.h"

__device__ bool intersectClosestSBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	float t_max = AT_MATH_INF,
	bool enableLod = false);

__device__ bool intersectCloserSBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max,
	bool enableLod = false);

__device__ bool intersectAnySBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	bool enableLod = false);

#ifndef __AT_DEBUG__
#include "kernel/sbvh_impl.cuh"
#endif
