#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

__device__ void addIntersectFuncs();

AT_DEVICE_API bool intersectShape(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt);

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::hitrecordOption* recOpt);
