#pragma once

#include "aten4idaten.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/intersecttype.h"

__device__ bool intersectClosestStacklessBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max = AT_MATH_INF);

__device__ bool intersectCloserStacklessBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max);

__device__ bool intersectAnyStacklessBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect);

#ifndef __AT_DEBUG__
#include "kernel/stackless_bvh_impl.cuh"
#endif
