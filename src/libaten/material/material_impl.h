#pragma once

#include <type_traits>

#include "material/material.h"
#include "material/sample_texture.h"
#include "material/emissive.h"
#include "material/lambert.h"
#include "material/oren_nayar.h"
#include "material/specular.h"
#include "material/refraction.h"
#include "material/blinn.h"
#include "material/ggx.h"
#include "material/beckman.h"
#include "material/velvet.h"
#include "material/lambert_refraction.h"
#include "material/microfacet_refraction.h"
#include "material/disney_brdf.h"
#include "material/retroreflective.h"
#include "material/car_paint.h"

namespace AT_NAME
{
    inline AT_DEVICE_API aten::vec4 material::sampleAlbedoMap(
        const aten::MaterialParameter* mtrl,
        real u, real v,
        uint32_t lod/*= 0*/)
    {
        return sampleTexture(mtrl->albedoMap, u, v, mtrl->baseColor, lod);
    }

    inline AT_DEVICE_API void material::sampleMaterial(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* mtrl,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
#ifdef __CUDACC__
        float u, float v)
#else
        float u, float v,
        bool is_light_path/*= false*/)
#endif
    {
#ifdef __CUDACC__
        constexpr bool is_light_path = true;
#endif
        switch (mtrl->type) {
        case aten::MaterialType::Emissive:
            AT_NAME::emissive::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Lambert:
            AT_NAME::lambert::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::OrneNayar:
            AT_NAME::OrenNayar::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Specular:
            AT_NAME::specular::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Refraction:
            AT_NAME::refraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Blinn:
            AT_NAME::MicrofacetBlinn::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::GGX:
            AT_NAME::MicrofacetGGX::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Beckman:
            AT_NAME::MicrofacetBeckman::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Velvet:
            AT_NAME::MicrofacetVelvet::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Lambert_Refraction:
            AT_NAME::LambertRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            AT_NAME::MicrofacetRefraction::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::Retroreflective:
            AT_NAME::Retroreflective::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        case aten::MaterialType::CarPaint:
            AT_NAME::CarPaint::sample(result, mtrl, normal, wi, orgnormal, sampler, pre_sampled_r, u, v, is_light_path);
            break;
        case aten::MaterialType::Disney:
            AT_NAME::DisneyBRDF::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        default:
            AT_ASSERT(false);
            AT_NAME::lambert::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, is_light_path);
            break;
        }
    }

    inline AT_DEVICE_API void material::sampleMaterialWithExternalAlbedo(
        AT_NAME::MaterialSampling* result,
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
            AT_NAME::MicrofacetGGX::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v, externalAlbedo);
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
        default:
            AT_ASSERT(false);
            AT_NAME::lambert::sample(result, mtrl, normal, wi, orgnormal, sampler, externalAlbedo, false);
            break;
        }
    }

    inline AT_DEVICE_API real material::samplePDF(
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
        default:
            AT_ASSERT(false);
            pdf = AT_NAME::lambert::pdf(normal, wo);
            break;
        }

        return pdf;
    }

    inline AT_DEVICE_API aten::vec3 material::sampleDirection(
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
        default:
            AT_ASSERT(false);
            return AT_NAME::lambert::sampleDirection(normal, sampler);
        }

        return aten::vec3(0, 1, 0);
    }

    inline AT_DEVICE_API aten::vec3 material::sampleBSDF(
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
        default:
            AT_ASSERT(false);
            return AT_NAME::lambert::bsdf(mtrl, u, v);
        }

        return aten::vec3();
    }

    inline AT_DEVICE_API aten::vec3 material::sampleBSDFWithExternalAlbedo(
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
        default:
            AT_ASSERT(false);
            return AT_NAME::lambert::bsdf(mtrl, externalAlbedo);
        }

        return aten::vec3();
    }

    inline AT_DEVICE_API real material::applyNormal(
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

    namespace detail {
        template <class T, class HasVariable = void>
        struct has_texture_variable : public std::false_type {};

        template <class T>
        struct has_texture_variable<
            T,
            std::void_t<decltype(T::textures)>> : public std::true_type {};
    }

    template <class CONTEXT>
    inline AT_DEVICE_API bool FillMaterial(
        aten::MaterialParameter& dst_mtrl,
        const CONTEXT& ctxt,
        const int32_t mtrl_id,
        const bool is_voxel)
    {
        bool is_valid_mtrl = mtrl_id >= 0;

        if (is_valid_mtrl) {
            if (is_voxel) {
                // Replace to lambert.
                const auto& albedo = ctxt.GetMaterial(static_cast<uint32_t>(mtrl_id)).baseColor;
                dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, aten::MaterialAttributeLambert);
                dst_mtrl.baseColor = albedo;
            }
            else {
                dst_mtrl = ctxt.GetMaterial(static_cast<uint32_t>(mtrl_id));
            }
            // Check if `context` class has `textures` variable.
            if constexpr (detail::has_texture_variable<CONTEXT>::value) {
                // Check if `context::textures` is `int32_t` type.
                static_assert(
                    std::is_integral_v<std::remove_pointer_t<decltype(CONTEXT::textures)>>,
                    "context::textures has to be integral");
                if constexpr (std::is_integral_v<std::remove_pointer_t<decltype(CONTEXT::textures)>>) {
                    dst_mtrl.albedoMap = (int)(dst_mtrl.albedoMap >= 0 ? ctxt.textures[dst_mtrl.albedoMap] : -1);
                    dst_mtrl.normalMap = (int)(dst_mtrl.normalMap >= 0 ? ctxt.textures[dst_mtrl.normalMap] : -1);
                    dst_mtrl.roughnessMap = (int)(dst_mtrl.roughnessMap >= 0 ? ctxt.textures[dst_mtrl.roughnessMap] : -1);
                }
            }
        }
        else {
            // TODO
            dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, aten::MaterialAttributeLambert);
            dst_mtrl.baseColor = aten::vec3(1.0f);
        }

        return is_valid_mtrl;
    }

}
