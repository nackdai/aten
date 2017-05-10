#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

AT_DEVICE_API bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec);
