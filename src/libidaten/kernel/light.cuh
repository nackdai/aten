#pragma once

#include "kernel/context.cuh"
#include "kernel/intersect.cuh"

#include "aten4idaten.h"

__device__ void sampleLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	const aten::vec3& normal,
	aten::sampler* sampler);

#ifndef __AT_DEBUG__
#include "kernel/light_impl.cuh"
#endif
