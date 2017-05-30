#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

__device__ void addIntersectFuncs();

__device__ bool intersectShape(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt);

__device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt);

__device__ bool hitTriangle(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& ray,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt);

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::hitrecordOption* recOpt);
