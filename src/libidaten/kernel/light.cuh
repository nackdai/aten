#pragma once

#include "kernel/context.cuh"
#include "kernel/intersect.cuh"

#include "aten4idaten.h"

__device__ void getTriangleSamplePosNormalArea(
	aten::hitable::SamplePosNormalPdfResult* result,
	Context* ctxt,
	const aten::ShapeParameter* shape,
	aten::sampler* sampler);

__device__  void sampleAreaLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler);

__device__ void sampleLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler);

#ifndef __AT_DEBUG__
#include "kernel/light_impl.cuh"
#endif
