#pragma once

#include "aten4idaten.h"

__device__ void addMaterialFuncs();

__device__ void sampleMaterial(
	AT_NAME::MaterialSampling* result,
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::hitrecord& hitrec,
	aten::sampler* sampler,
	float u, float v);
