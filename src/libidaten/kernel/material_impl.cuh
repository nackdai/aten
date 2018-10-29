#include "kernel/material.cuh"

#include "kernel/idatendefs.cuh"

//////////////////////////////////////////////////////////

// For layered material....

AT_CUDA_INLINE __device__ void sampleLayerMaterial(
    AT_NAME::MaterialSampling* result,
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    float u, float v)
{
    real weight = 1;

    for (int i = 0; i < AT_COUNTOF(mtrl->layer); i++) {
        if (mtrl->layer[i] < 0) {
            break;
        }

        auto* param = &ctxt->mtrls[mtrl->layer[i]];

        aten::vec3 appliedNml = normal;

        // NOTE
        // 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
        if (i > 0) {
            auto normalMap = (int)(param->normalMap >= 0 ? ctxt->textures[param->normalMap] : -1);
            AT_NAME::applyNormalMap(normalMap, normal, appliedNml, u, v);
        }

        AT_NAME::MaterialSampling sampleres;
        sampleMaterial(
            &sampleres,
            ctxt,
            param,
            appliedNml,
            wi,
            orgnormal,
            sampler,
            u, v);

        const auto f = aten::clamp<real>(sampleres.fresnel, 0, 1);

        result->pdf += weight * f * sampleres.pdf;

        // bsdf includes fresnale value.
        result->bsdf += weight * sampleres.bsdf;
        //ret.bsdf += weight * f * sampleres.bsdf;

        // TODO
        // ret.fresnel

        weight = aten::clamp<real>(weight - f, 0, 1);
        if (weight <= 0) {
            break;
        }

        if (i == 0) {
            result->dir = sampleres.dir;
        }
    }
}

AT_CUDA_INLINE __device__ real sampleLayerPDF(
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v)
{
    real pdf = 0;

    real weight = 1;
    real ior = 1;    // 真空から始める.

    for (int i = 0; i < AT_COUNTOF(mtrl->layer); i++) {
        if (mtrl->layer[i] < 0) {
            break;
        }

        auto* param = &ctxt->mtrls[mtrl->layer[i]];

        aten::vec3 appliedNml = normal;

        // NOTE
        // 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
        if (i > 0) {
            auto normalMap = (int)(param->normalMap >= 0 ? ctxt->textures[param->normalMap] : -1);
            AT_NAME::applyNormalMap(normalMap, normal, appliedNml, u, v);
        }

        auto p = samplePDF(ctxt, param, appliedNml, wi, wo, u, v);
        auto f = computeFresnel(param, appliedNml, wi, wo, ior);

        f = aten::clamp<real>(f, 0, 1);

        pdf += weight * p;

        weight = aten::clamp<real>(weight - f, 0, 1);
        if (weight <= 0) {
            break;
        }

        // 上層の値を下層に使う.
        ior = param->ior;
    }

    return pdf;
}

AT_CUDA_INLINE __device__ aten::vec3 sampleLayerDirection(
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    real u, real v,
    aten::sampler* sampler)
{
    auto* param = &ctxt->mtrls[mtrl->layer[0]];

    auto dir = sampleDirection(ctxt, param, normal, wi, u, v, sampler);

    return std::move(dir);
}

AT_CUDA_INLINE __device__ aten::vec3 sampleLayerBSDF(
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v)
{
    aten::vec3 bsdf;

    real weight = 1;
    real ior = 1;    // 真空から始める.

    for (int i = 0; i < AT_COUNTOF(mtrl->layer); i++) {
        if (mtrl->layer[i] < 0) {
            break;
        }

        auto* param = &ctxt->mtrls[mtrl->layer[i]];

        aten::vec3 appliedNml = normal;

        // NOTE
        // 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
        if (i > 0) {
            auto normalMap = (int)(param->normalMap >= 0 ? ctxt->textures[param->normalMap] : -1);
            AT_NAME::applyNormalMap(normalMap, normal, appliedNml, u, v);
        }

        auto b = sampleBSDF(ctxt, param, appliedNml, wi, wo, u, v);
        auto f = computeFresnel(param, appliedNml, wi, wo, ior);

        f = aten::clamp<real>(f, 0, 1);

        // bsdf includes fresnel value.
        bsdf += weight * b;
        //bsdf += weight * f * b;

        weight = aten::clamp<real>(weight - f, 0, 1);
        if (weight <= 0) {
            break;
        }

        // 上層の値を下層に使う.
        ior = param->ior;
    }

    return std::move(bsdf);
}

//////////////////////////////////////////////////////////

AT_CUDA_INLINE __device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const Context* ctxt,
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
    case aten::MaterialType::Velvet:
        AT_NAME::MicrofacetVelvet::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Lambert_Refraction:
        AT_NAME::LambertRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Microfacet_Refraction:
        AT_NAME::MicrofacetRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Disney:
        AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::CarPaint:
        AT_NAME::CarPaintBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Layer:
        sampleLayerMaterial(result, ctxt, mtrl, normal, wi, orgnormal, sampler, u, v);
        break;
    case aten::MaterialType::Toon:
        break;
    }
}

AT_CUDA_INLINE __device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    float u, float v,
    const aten::vec3& externalAlbedo)
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
    case aten::MaterialType::Disney:
        AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::CarPaint:
        AT_NAME::CarPaintBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
        break;
    case aten::MaterialType::Layer:
        sampleLayerMaterial(result, ctxt, mtrl, normal, wi, orgnormal, sampler, u, v);
        break;
    case aten::MaterialType::Toon:
        break;
    }
}

AT_CUDA_INLINE __device__ real samplePDF(
    const Context* ctxt,
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
    case aten::MaterialType::Disney:
        pdf = AT_NAME::DisneyBRDF::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::CarPaint:
        pdf = AT_NAME::CarPaintBRDF::pdf(mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::Layer:
        pdf = sampleLayerPDF(ctxt, mtrl, normal, wi, wo, u, v);
        break;
    case aten::MaterialType::Toon:
        break;
    }

    return pdf;
}

AT_CUDA_INLINE __device__ aten::vec3 sampleDirection(
    const Context* ctxt,
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
    case aten::MaterialType::Velvet:
        return AT_NAME::MicrofacetVelvet::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Lambert_Refraction:
        return AT_NAME::LambertRefraction::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Microfacet_Refraction:
        return AT_NAME::MicrofacetRefraction::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaintBRDF::sampleDirection(mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Layer:
        return sampleLayerDirection(ctxt, mtrl, normal, wi, u, v, sampler);
    case aten::MaterialType::Toon:
        break;
    }

    return std::move(aten::vec3(0, 1, 0));
}

AT_CUDA_INLINE __device__ aten::vec3 sampleBSDF(
    const Context* ctxt,
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
    case aten::MaterialType::Velvet:
        return AT_NAME::MicrofacetVelvet::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Lambert_Refraction:
        return AT_NAME::LambertRefraction::bsdf(mtrl, u, v);
    case aten::MaterialType::Microfacet_Refraction:
        return AT_NAME::MicrofacetRefraction::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaintBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Layer:
        return sampleLayerBSDF(ctxt, mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Toon:
        break;
    }

    return std::move(aten::vec3());
}

AT_CUDA_INLINE __device__ aten::vec3 sampleBSDF(
    const Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    const aten::vec3& externalAlbedo)
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
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::CarPaint:
        return AT_NAME::CarPaintBRDF::bsdf(mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Layer:
        return sampleLayerBSDF(ctxt, mtrl, normal, wi, wo, u, v);
    case aten::MaterialType::Toon:
        break;
    }

    return std::move(aten::vec3());
}

AT_CUDA_INLINE __device__ real computeFresnel(
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real outsideIor/*= 1*/)
{
    switch (mtrl->type) {
    case aten::MaterialType::Blinn:
    case aten::MaterialType::GGX:
    case aten::MaterialType::Beckman:
        return AT_NAME::material::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Emissive:
        return AT_NAME::emissive::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Lambert:
        return AT_NAME::lambert::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::OrneNayar:
        return AT_NAME::OrenNayar::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Specular:
        return AT_NAME::specular::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Refraction:
        return AT_NAME::refraction::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Disney:
        return AT_NAME::DisneyBRDF::computeFresnel(mtrl, normal, wi, wo, outsideIor);
    case aten::MaterialType::Layer:
    case aten::MaterialType::Toon:
        return real(1);
    }

    return real(1);
}