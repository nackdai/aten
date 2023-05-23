#pragma once

#include "kernel/device_scene_context.cuh"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

__device__ bool hitTriangle(
    const aten::TriangleParameter* prim,
    const idaten::context* ctxt,
    const aten::ray& ray,
    aten::Intersection* isect);

__device__ int32_t hit4Triangles1Ray(
    const idaten::context* ctxt,
    float4 primIdx, int32_t num,
    float4* resultT,
    float4* resultA,
    float4* resultB,
    aten::vec3 org, aten::vec3 dir,
    float4 v0x, float4 v0y, float4 v0z,
    float4 e1x, float4 e1y, float4 e1z);

__device__ bool hitAABB(
    aten::vec3 org,
    aten::vec3 dir,
    float4 boxmin, float4 boxmax,
    real t_min, real t_max,
    real* t_result);

__device__ bool hitAABB(
    aten::vec3 org,
    aten::vec3 dir,
    float4 boxmin, float4 boxmax,
    real t_min, real t_max,
    real* t_result,
    aten::vec3* nml);

__device__ int32_t hit4AABBWith1Ray(
    aten::vec4* result,
    const aten::vec3& org,
    const aten::vec3& dir,
    const float4& bminx, const float4& bmaxx,
    const float4& bminy, const float4& bmaxy,
    const float4& bminz, const float4& bmaxz,
    float t_min, float t_max);

#ifndef __AT_DEBUG__
#include "kernel/intersect_impl.cuh"
#endif
