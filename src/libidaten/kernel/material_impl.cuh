#include "kernel/material.cuh"

#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ void sampleMaterial(
	AT_NAME::MaterialSampling* result,
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	float u, float v)
{
	switch (mtrl->type) {
	case aten::MaterialType::Emissive:
		AT_NAME::emissive::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Lambert:
		AT_NAME::lambert::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::OrneNayar:
		AT_NAME::OrenNayar::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Specular:
		AT_NAME::specular::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Refraction:
		AT_NAME::refraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Blinn:
		AT_NAME::MicrofacetBlinn::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::GGX:
		AT_NAME::MicrofacetGGX::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Beckman:
		AT_NAME::MicrofacetBeckman::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Disney:
		AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
		break;
	case aten::MaterialType::Toon:
	case aten::MaterialType::Layer:
		break;
	}
}

AT_CUDA_INLINE __device__ real samplePDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	real pdf = real(0);

	switch (mtrl->type) {
	case aten::MaterialType::Emissive:
		pdf = AT_NAME::emissive::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Lambert:
		pdf = AT_NAME::lambert::pdf(normal, wo);
		break;
	case aten::MaterialType::OrneNayar:
		pdf = AT_NAME::OrenNayar::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Specular:
		pdf = AT_NAME::specular::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Refraction:
		pdf = AT_NAME::refraction::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Blinn:
		pdf = AT_NAME::MicrofacetBlinn::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::GGX:
		pdf = AT_NAME::MicrofacetGGX::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Beckman:
		pdf = AT_NAME::MicrofacetBeckman::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Disney:
		pdf = AT_NAME::DisneyBRDF::pdf(mtrl, normal, wi, wo, u, v);
		break;
	case aten::MaterialType::Toon:
	case aten::MaterialType::Layer:
		break;
	}

	return pdf;
}

AT_CUDA_INLINE __device__ aten::vec3 sampleDirection(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	real u, real v,
	aten::sampler* sampler)
{
	switch (mtrl->type) {
	case aten::MaterialType::Emissive:
		return AT_NAME::emissive::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Lambert:
		return AT_NAME::lambert::sampleDirection(normal, sampler);
	case aten::MaterialType::OrneNayar:
		return AT_NAME::OrenNayar::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Specular:
		return AT_NAME::specular::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Refraction:
		return AT_NAME::refraction::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Blinn:
		return AT_NAME::MicrofacetBlinn::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::GGX:
		return AT_NAME::MicrofacetGGX::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Beckman:
		return AT_NAME::MicrofacetBeckman::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Disney:
		return AT_NAME::DisneyBRDF::sampleDirection(mtrl, normal, wi, u, v, sampler);
	case aten::MaterialType::Toon:
	case aten::MaterialType::Layer:
		break;
	}

	return std::move(aten::vec3(0, 1, 0));
}

AT_CUDA_INLINE __device__ aten::vec3 sampleBSDF(
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& wo,
	real u, real v)
{
	switch (mtrl->type) {
	case aten::MaterialType::Emissive:
		return AT_NAME::emissive::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Lambert:
		return AT_NAME::lambert::bsdf(mtrl, u, v);
	case aten::MaterialType::OrneNayar:
		return AT_NAME::OrenNayar::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Specular:
		return AT_NAME::specular::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Refraction:
		return AT_NAME::refraction::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Blinn:
		return AT_NAME::MicrofacetBlinn::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::GGX:
		return AT_NAME::MicrofacetGGX::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Beckman:
		return AT_NAME::MicrofacetBeckman::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Disney:
		return AT_NAME::DisneyBRDF::bsdf(mtrl, normal, wi, wo, u, v);
	case aten::MaterialType::Toon:
	case aten::MaterialType::Layer:
		break;
	}

	return std::move(aten::vec3());
}
