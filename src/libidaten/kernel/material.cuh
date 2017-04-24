#pragma once

#include "aten4idaten.h"

__device__ AT_NAME::MaterialSampling sampleMaterial(
	const aten::MaterialParameter& mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::hitrecord& hitrec,
	aten::sampler* sampler,
	float u, float v);
