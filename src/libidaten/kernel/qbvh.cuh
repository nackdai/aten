#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"
#include "cuda/cudadefs.h"
#include "kernel/intersect.cuh"
#include "cuda/helper_math.h"
#include "kernel/intersecttype.h"

__device__ bool intersectQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max = AT_MATH_INF);

__device__ bool intersectCloserQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max);

__device__ bool intersectAnyQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect);

#ifndef __AT_DEBUG__
#include "kernel/qbvh_impl.cuh"
#endif
