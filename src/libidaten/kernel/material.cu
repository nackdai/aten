#include "kernel/material.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

typedef void(*FuncSampleMaterial)(
	AT_NAME::MaterialSampling*,
	const aten::MaterialParameter*,
	const aten::vec3&,
	const aten::vec3&,
	const aten::vec3&,
	aten::sampler*,
	float, float,
	bool);

typedef real(*FuncPDF)(
	const aten::MaterialParameter*,
	const aten::vec3&,
	const aten::vec3&,
	const aten::vec3&,
	real, real);

typedef aten::vec3(*FuncSampleDirection)(
	const aten::MaterialParameter*,
	const aten::vec3&,
	const aten::vec3&,
	real u, real v,
	aten::sampler*);

typedef aten::vec3(*FuncBSDF)(
	const aten::MaterialParameter*,
	const aten::vec3&,
	const aten::vec3&,
	const aten::vec3&,
	real, real);

__device__ void sampleMtrlNotSupported(
	AT_NAME::MaterialSampling*,
	const aten::MaterialParameter* param,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	real u, real v,
	bool isLightPath)
{
	printf("Sample Material Not Supported[%d]\n", param->type);
}

__device__ real pdfNotSupported(
	const aten::MaterialParameter* param,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	printf("Sample PDF Not Supported[%d]\n", param->type);
	return 0;
}

__device__ aten::vec3 sampleDirectionNotSupported(
	const aten::MaterialParameter* param,
	const aten::vec3& normal,
	const aten::vec3& wi,
	real u, real v,
	aten::sampler* sampler)
{
	printf("Sample SampleDirection Not Supported[%d]\n", param->type);
	return aten::vec3();
}

__device__ aten::vec3 bsdfNotSupported(
	const aten::MaterialParameter* param,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	printf("Sample BSDF Not Supported[%d]\n", param->type);
	return aten::vec3();
}

__device__ FuncSampleMaterial funcSampleMaterial[aten::MaterialType::MaterialTypeMax];
__device__ FuncPDF funcPDF[aten::MaterialType::MaterialTypeMax];
__device__ FuncSampleDirection funcSampleDirection[aten::MaterialType::MaterialTypeMax];
__device__ FuncBSDF funcBSDF[aten::MaterialType::MaterialTypeMax];

__device__ void addMaterialFuncs()
{
#if 1
	funcSampleMaterial[aten::MaterialType::Emissive] = AT_NAME::emissive::sample;			// Emissive
	funcSampleMaterial[aten::MaterialType::Lambert] = AT_NAME::lambert::sample;				// Lambert
	funcSampleMaterial[aten::MaterialType::OrneNayar] = AT_NAME::OrenNayar::sample;			// OrneNayar
	funcSampleMaterial[aten::MaterialType::Specular] = AT_NAME::specular::sample;			// Specular
	funcSampleMaterial[aten::MaterialType::Refraction] = AT_NAME::refraction::sample;		// Refraction
	funcSampleMaterial[aten::MaterialType::Blinn] = AT_NAME::MicrofacetBlinn::sample;		// MicrofacetBlinn
	funcSampleMaterial[aten::MaterialType::GGX] = AT_NAME::MicrofacetGGX::sample;			// MicrofacetGGX
	funcSampleMaterial[aten::MaterialType::Beckman] = AT_NAME::MicrofacetBeckman::sample;	// MicrofacetBeckman
	funcSampleMaterial[aten::MaterialType::Disney] = sampleMtrlNotSupported;	// DisneyBRDF
	funcSampleMaterial[aten::MaterialType::Toon] = sampleMtrlNotSupported;		// Toon
	funcSampleMaterial[aten::MaterialType::Layer] = sampleMtrlNotSupported;		// Layer
#else
	funcSampleMaterial[aten::MaterialType::Emissive] = sampleMtrlNotSupported;			// Emissive
	funcSampleMaterial[aten::MaterialType::Lambert] = sampleMtrlNotSupported;			// Lambert
	funcSampleMaterial[aten::MaterialType::OrneNayar] = sampleMtrlNotSupported;			// OrneNayar
	funcSampleMaterial[aten::MaterialType::Specular] = AT_NAME::specular::sample;		// Specular
	funcSampleMaterial[aten::MaterialType::Refraction] = AT_NAME::refraction::sample;	// Refraction
	funcSampleMaterial[aten::MaterialType::Blinn] = sampleMtrlNotSupported;		// MicrofacetBlinn
	funcSampleMaterial[aten::MaterialType::GGX] = sampleMtrlNotSupported;		// MicrofacetGGX
	funcSampleMaterial[aten::MaterialType::Beckman] = sampleMtrlNotSupported;	// MicrofacetBeckman
	funcSampleMaterial[aten::MaterialType::Disney] = sampleMtrlNotSupported;	// DisneyBRDF
	funcSampleMaterial[aten::MaterialType::Toon] = sampleMtrlNotSupported;		// Toon
	funcSampleMaterial[aten::MaterialType::Layer] = sampleMtrlNotSupported;		// Layer
#endif

	funcPDF[aten::MaterialType::Emissive] = AT_NAME::emissive::pdf;			// Emissive
	funcPDF[aten::MaterialType::Lambert] = AT_NAME::lambert::pdf;			// Lambert
	funcPDF[aten::MaterialType::OrneNayar] = AT_NAME::OrenNayar::pdf;		// OrneNayar
	funcPDF[aten::MaterialType::Specular] = AT_NAME::specular::pdf;			// Specular
	funcPDF[aten::MaterialType::Refraction] = AT_NAME::refraction::pdf;		// Refraction
	funcPDF[aten::MaterialType::Blinn] = AT_NAME::MicrofacetBlinn::pdf;		// MicrofacetBlinn
	funcPDF[aten::MaterialType::GGX] = AT_NAME::MicrofacetGGX::pdf;			// MicrofacetGGX
	funcPDF[aten::MaterialType::Beckman] = AT_NAME::MicrofacetBeckman::pdf;	// MicrofacetBeckman
	funcPDF[aten::MaterialType::Disney] = pdfNotSupported;	// DisneyBRDF
	funcPDF[aten::MaterialType::Toon] = pdfNotSupported;	// Toon
	funcPDF[aten::MaterialType::Layer] = pdfNotSupported;	// Layer

	funcSampleDirection[aten::MaterialType::Emissive] = AT_NAME::emissive::sampleDirection;			// Emissive
	funcSampleDirection[aten::MaterialType::Lambert] = AT_NAME::lambert::sampleDirection;			// Lambert
	funcSampleDirection[aten::MaterialType::OrneNayar] = AT_NAME::OrenNayar::sampleDirection;		// OrneNayar
	funcSampleDirection[aten::MaterialType::Specular] = AT_NAME::specular::sampleDirection;			// Specular
	funcSampleDirection[aten::MaterialType::Refraction] = AT_NAME::refraction::sampleDirection;		// Refraction
	funcSampleDirection[aten::MaterialType::Blinn] = AT_NAME::MicrofacetBlinn::sampleDirection;		// MicrofacetBlinn
	funcSampleDirection[aten::MaterialType::GGX] = AT_NAME::MicrofacetGGX::sampleDirection;			// MicrofacetGGX
	funcSampleDirection[aten::MaterialType::Beckman] = AT_NAME::MicrofacetBeckman::sampleDirection;	// MicrofacetBeckman
	funcSampleDirection[aten::MaterialType::Disney] = sampleDirectionNotSupported;	// DisneyBRDF
	funcSampleDirection[aten::MaterialType::Toon] = sampleDirectionNotSupported;	// Toon
	funcSampleDirection[aten::MaterialType::Layer] = sampleDirectionNotSupported;	// Layer

	funcBSDF[aten::MaterialType::Emissive] = AT_NAME::emissive::bsdf;			// Emissive
	funcBSDF[aten::MaterialType::Lambert] = AT_NAME::lambert::bsdf;				// Lambert
	funcBSDF[aten::MaterialType::OrneNayar] = AT_NAME::OrenNayar::bsdf;			// OrneNayar
	funcBSDF[aten::MaterialType::Specular] = AT_NAME::specular::bsdf;			// Specular
	funcBSDF[aten::MaterialType::Refraction] = AT_NAME::refraction::bsdf;		// Refraction
	funcBSDF[aten::MaterialType::Blinn] = AT_NAME::MicrofacetBlinn::bsdf;		// MicrofacetBlinn
	funcBSDF[aten::MaterialType::GGX] = AT_NAME::MicrofacetGGX::bsdf;			// MicrofacetGGX
	funcBSDF[aten::MaterialType::Beckman] = AT_NAME::MicrofacetBeckman::bsdf;	// MicrofacetBeckman
	funcBSDF[aten::MaterialType::Disney] = bsdfNotSupported;	// DisneyBRDF
	funcBSDF[aten::MaterialType::Toon] = bsdfNotSupported;		// Toon
	funcBSDF[aten::MaterialType::Layer] = bsdfNotSupported;		// Layer
}

__device__ void sampleMaterial(
	AT_NAME::MaterialSampling* result,
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	float u, float v)
{
	funcSampleMaterial[mtrl->type](result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
}

__device__ real samplePDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	return funcPDF[mtrl->type](mtrl, normal, wi, wo, u, v);
}

__device__ aten::vec3 sampleDirection(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	real u, real v,
	aten::sampler* sampler)
{
	return funcSampleDirection[mtrl->type](mtrl, normal, wi, u, v, sampler);
}

__device__ aten::vec3 sampleBSDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	return funcBSDF[mtrl->type](mtrl, normal, wi, wo, u, v);
}
