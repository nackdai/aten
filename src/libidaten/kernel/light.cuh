#pragma once

#include "kernel/context.cuh"
#include "aten4idaten.h"

__device__ void addLighFuncs();

__device__ void sampleLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler);
