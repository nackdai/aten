#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

__device__ void addIntersectFuncs();

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

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect);
