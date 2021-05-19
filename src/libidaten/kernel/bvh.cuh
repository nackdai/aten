#pragma once

#include "aten4idaten.h"
#include "cuda/cudadefs.h"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "cuda/helper_math.h"
#include "kernel/intersecttype.h"

__device__ bool intersectBVH(
    const Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    float t_max = AT_MATH_INF,
    bool enableLod = false,
    int depth = -1);

__device__ bool intersectCloserBVH(
    const Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max,
    bool enableLod = false,
    int depth = -1);

__device__ bool intersectAnyBVH(
    const Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    bool enableLod = false,
    int depth = -1);

#ifndef __AT_DEBUG__
#include "kernel/bvh_impl.cuh"
#endif
