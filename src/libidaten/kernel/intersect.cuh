#pragma once

#include "kernel/context.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "aten4idaten.h"

__device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect);

__device__ bool hitTriangle(
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& ray,
	aten::Intersection* isect);

__device__ bool hitAABB(
	aten::vec3 org,
	aten::vec3 dir,
	float4 boxmin, float4 boxmax,
	real t_min, real t_max);

__device__ bool hitAABB(
	aten::vec3 org,
	aten::vec3 dir,
	float4 boxmin, float4 boxmax,
	real t_min, real t_max,
	real* t_result);

__device__ void evalHitResultTriangle(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect);

__device__ void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect);

#ifndef __AT_DEBUG__
#include "kernel/intersect_impl.cuh"
#endif
