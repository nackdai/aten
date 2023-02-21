#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/intersecttype.h"

__device__ bool intersectClosestSBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    float t_max = AT_MATH_INF,
    bool enableLod = false,
    int32_t depth = -1);

__device__ bool intersectCloserSBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max,
    bool enableLod = false,
    int32_t depth = -1);

__device__ bool intersectAnySBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    bool enableLod = false,
    int32_t depth = -1);

#ifndef __AT_DEBUG__
#include "kernel/sbvh_impl.cuh"
#endif
