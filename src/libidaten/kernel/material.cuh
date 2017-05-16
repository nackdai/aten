#pragma once

#include "aten4idaten.h"

__device__ void addMaterialFuncs();

__device__ void sampleMaterial(
	AT_NAME::MaterialSampling* result,
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	float u, float v);

__device__ real samplePDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v);

__device__ aten::vec3 sampleDirection(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	real u, real v,
	aten::sampler* sampler);
	
__device__ aten::vec3 sampleBSDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v);
