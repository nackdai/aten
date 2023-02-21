#include "kernel/material.cuh"

#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    real pre_sampled_r,
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
    case aten::MaterialType::Velvet:
        AT_NAME::MicrofacetVelvet::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Lambert_Refraction:
        AT_NAME::LambertRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Microfacet_Refraction:
        AT_NAME::MicrofacetRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Retroreflective:
        AT_NAME::Retroreflective::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::CarPaint:
        AT_NAME::CarPaint::sample(result, mtrl, normal, wi, orgnormal, sampler, pre_sampled_r, u, v, false);
        break;
    case aten::MaterialType::Disney:
        AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    }
}

AT_CUDA_INLINE __device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    real pre_sampled_r,
    float u, float v,
    const aten::vec4& externalAlbedo)
{
    switch (mtrl->type) {
    case aten::MaterialType::Emissive:
        AT_NAME::emissive::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Lambert:
        AT_NAME::lambert::sample(result, mtrl, normal, wi, orgnormal, sampler, externalAlbedo, false);
        break;
    case aten::MaterialType::OrneNayar:
        AT_NAME::OrenNayar::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Specular:
        AT_NAME::specular::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Refraction:
        // TODO
        AT_NAME::refraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Blinn:
        AT_NAME::MicrofacetBlinn::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::GGX:
        AT_NAME::MicrofacetGGX::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Beckman:
        AT_NAME::MicrofacetBeckman::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Velvet:
        AT_NAME::MicrofacetVelvet::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Lambert_Refraction:
        AT_NAME::LambertRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, externalAlbedo, false);
        break;
    case aten::MaterialType::Microfacet_Refraction:
        AT_NAME::MicrofacetRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Retroreflective:
        AT_NAME::Retroreflective::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::CarPaint:
        AT_NAME::CarPaint::sample(result, mtrl, normal, wi, orgnormal, sampler, pre_sampled_r, u, v, externalAlbedo, false);
        break;
    case aten::MaterialType::Disney:
        AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    }
}

AT_CUDA_INLINE __device__ real samplePDF(
    const idaten::Context* ctxt,
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
    case aten::MaterialType::Velvet:
        pdf = AT_NAME::MicrofacetVelvet::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::Lambert_Refraction:
        pdf = AT_NAME::LambertRefraction::pdf(normal, wo);
        break;
    case aten::MaterialType::Microfacet_Refraction:
        pdf = AT_NAME::MicrofacetRefraction::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::Retroreflective:
        pdf = AT_NAME::Retroreflective::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::CarPaint:
        pdf = AT_NAME::CarPaint::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::Disney:
        pdf = AT_NAME::DisneyBRDF::pdf(mtrl, normal, wi, wo, u, v);
        break;
    }

    return pdf;
}

AT_CUDA_INLINE __device__ aten::vec3 sampleDirection(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    real u, real v,
    aten::sampler* sampler,
    real pre_sampled_r)
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
    case aten::MaterialType::Velvet:
        return AT_NAME::MicrofacetVelvet::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Lambert_Refraction:
        return AT_NAME::LambertRefraction::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Microfacet_Refraction:
        return AT_NAME::MicrofacetRefraction::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Retroreflective:
        return AT_NAME::Retroreflective::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaint::sampleDirection(mtrl, normal, wi, u, v, sampler, pre_sampled_r);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::sampleDirection(mtrl, normal, wi, u, v, sampler);
    }

    return aten::vec3(0, 1, 0);
}

AT_CUDA_INLINE __device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    real pre_sampled_r)
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
    case aten::MaterialType::Velvet:
        return AT_NAME::MicrofacetVelvet::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Lambert_Refraction:
        return AT_NAME::LambertRefraction::bsdf(mtrl, u, v);
    case aten::MaterialType::Microfacet_Refraction:
        return AT_NAME::MicrofacetRefraction::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Retroreflective:
        return AT_NAME::Retroreflective::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaint::bsdf(mtrl, normal, wi, wo, u, v, pre_sampled_r);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    }

    return aten::vec3();
}

AT_CUDA_INLINE __device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    const aten::vec4& externalAlbedo,
    real pre_sampled_r)
{
    switch (mtrl->type) {
    case aten::MaterialType::Emissive:
        return AT_NAME::emissive::bsdf(mtrl, externalAlbedo);
    case aten::MaterialType::Lambert:
        return AT_NAME::lambert::bsdf(mtrl, externalAlbedo);
    case aten::MaterialType::OrneNayar:
        return AT_NAME::OrenNayar::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Specular:
        return AT_NAME::specular::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Refraction:
        return AT_NAME::refraction::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Blinn:
        return AT_NAME::MicrofacetBlinn::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::GGX:
        return AT_NAME::MicrofacetGGX::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Beckman:
        return AT_NAME::MicrofacetBeckman::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Velvet:
        return AT_NAME::MicrofacetVelvet::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Lambert_Refraction:
        return AT_NAME::LambertRefraction::bsdf(mtrl, externalAlbedo);
    case aten::MaterialType::Microfacet_Refraction:
        return AT_NAME::MicrofacetRefraction::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::Retroreflective:
        return AT_NAME::Retroreflective::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaint::bsdf(mtrl, normal, wi, wo, u, v, externalAlbedo, pre_sampled_r);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    }

    return aten::vec3();
}

AT_CUDA_INLINE __device__ real applyNormal(
    const aten::MaterialParameter* mtrl,
    const int32_t normalMapIdx,
    const aten::vec3& orgNml,
    aten::vec3& newNml,
    real u, real v,
    const aten::vec3& wi,
    aten::sampler* sampler)
{
    if (mtrl->type == aten::MaterialType::CarPaint) {
        return AT_NAME::CarPaint::applyNormalMap(
            mtrl,
            orgNml, newNml,
            u, v,
            wi,
            sampler);
    }
    else {
        AT_NAME::applyNormalMap(normalMapIdx, orgNml, newNml, u, v);
        return real(-1);
    }
}

AT_CUDA_INLINE __device__ real computeFresnel(
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real outsideIor/*= 1*/)
{
    // TODO

    return real(1);
}
