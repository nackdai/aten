#pragma once

#include "aten.h"

__device__ aten::LightSampleResult sampleLight(
	const aten::LightParameter& light,
	const aten::vec3& org,
	aten::sampler* sampler);
