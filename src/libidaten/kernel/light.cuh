#pragma once

#include "aten4idaten.h"

__device__ void addLighFuncs();

__device__ void sampleLight(
	aten::LightSampleResult* result,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler);
