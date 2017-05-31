#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

__device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect);
