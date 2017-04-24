#include "kernel/material.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

typedef AT_NAME::MaterialSampling(*FuncSampleMaterial)(
	const aten::MaterialParameter&,
	const aten::vec3&,
	const aten::vec3&,
	const aten::hitrecord&,
	aten::sampler*,
	float, float,
	bool);

__device__ AT_NAME::MaterialSampling sampleMaterial(
	const aten::MaterialParameter& mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::hitrecord& hitrec,
	aten::sampler* sampler,
	float u, float v)
{
	constexpr FuncSampleMaterial funcs[] = {
		AT_NAME::emissive::sample,			// Emissive
		AT_NAME::lambert::sample,			// Lambert
		AT_NAME::OrenNayar::sample,			// OrneNayar
		AT_NAME::specular::sample,			// Specular
		AT_NAME::refraction::sample,		// Refraction
		AT_NAME::MicrofacetBlinn::sample,	// MicrofacetBlinn
		AT_NAME::MicrofacetGGX::sample,		// MicrofacetGGX
		AT_NAME::MicrofacetBeckman::sample,	// MicrofacetBeckman
		nullptr,	// DisneyBRDF
		nullptr,	// Toon
		nullptr,	// Layer
	};

	auto ret = funcs[mtrl.type](mtrl, normal, wi, hitrec, sampler, u, v, false);

	return std::move(ret);
}