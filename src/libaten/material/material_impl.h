#pragma once

#include <type_traits>

#include "material/material.h"
#include "material/sample_texture.h"
#include "material/emissive.h"
#include "material/diffuse.h"
#include "material/oren_nayar.h"
#include "material/specular.h"
#include "material/refraction.h"
#include "material/ggx.h"
#include "material/beckman.h"
#include "material/velvet.h"
#include "material/microfacet_refraction.h"
#include "material/disney_brdf.h"
#include "material/retroreflective.h"
#include "material/toon_specular.h"
#include "material/car_paint.h"

namespace AT_NAME
{
    inline AT_DEVICE_API aten::vec4 material::sampleAlbedoMap(
        const aten::MaterialParameter* mtrl,
        float u, float v,
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
        float pre_sampled_r,
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
        case aten::MaterialType::Diffuse:
            AT_NAME::Diffuse::sample(result, mtrl, normal, wi, sampler);
            break;
        case aten::MaterialType::OrneNayar:
            AT_NAME::OrenNayar::sample(result, mtrl, normal, wi, sampler, u, v);
            break;
        case aten::MaterialType::Specular:
            AT_NAME::specular::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v);
            break;
        case aten::MaterialType::Refraction:
            AT_NAME::refraction::sample(result, mtrl, normal, wi, orgnormal, sampler);
            break;
        case aten::MaterialType::GGX:
            AT_NAME::MicrofacetGGX::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v);
            break;
        case aten::MaterialType::Beckman:
            AT_NAME::MicrofacetBeckman::sample(result, mtrl, normal, wi, orgnormal, sampler, u, v);
            break;
        case aten::MaterialType::Velvet:
            AT_NAME::MicrofacetVelvet::sample(result, mtrl, normal, wi, sampler, u, v);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            AT_NAME::MicrofacetRefraction::sample(*result, *mtrl, normal, wi, sampler, u, v);
            break;
        case aten::MaterialType::Retroreflective:
            AT_NAME::Retroreflective::sample(*result, *mtrl, normal, wi, sampler, u, v);
            break;
        case aten::MaterialType::CarPaint:
            AT_NAME::CarPaint::sample(result, mtrl, normal, wi, orgnormal, sampler, pre_sampled_r, u, v, is_light_path);
            break;
        case aten::MaterialType::Disney:
            AT_NAME::DisneyBRDF::sample(*result, *mtrl, normal, wi, sampler, u, v);
            break;
        default:
            AT_ASSERT(false);
            AT_NAME::Diffuse::sample(result, mtrl, normal, wi, sampler);
            break;
        }
    }

    inline AT_DEVICE_API float material::samplePDF(
        const aten::MaterialParameter* mtrl,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        float pdf = float(0);

        switch (mtrl->type) {
        case aten::MaterialType::Emissive:
            pdf = AT_NAME::emissive::pdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Diffuse:
            pdf = AT_NAME::Diffuse::pdf(normal, wo);
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
        case aten::MaterialType::GGX:
            pdf = AT_NAME::MicrofacetGGX::pdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Beckman:
            pdf = AT_NAME::MicrofacetBeckman::pdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Velvet:
            pdf = AT_NAME::MicrofacetVelvet::pdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            pdf = AT_NAME::MicrofacetRefraction::pdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Retroreflective:
            pdf = AT_NAME::Retroreflective::pdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::CarPaint:
            pdf = AT_NAME::CarPaint::pdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Disney:
            pdf = AT_NAME::DisneyBRDF::pdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::ToonSpecular:
            pdf = AT_NAME::ToonSpecular::ComputePDF(*mtrl, normal, wi, wo, u, v);
            break;
        default:
            AT_ASSERT(false);
            pdf = AT_NAME::Diffuse::pdf(normal, wo);
            break;
        }

        return pdf;
    }

    inline AT_DEVICE_API AT_NAME::MaterialSampling material::sampleBSDF(
        const aten::MaterialParameter* mtrl,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v,
        float pre_sampled_r)
    {
        AT_NAME::MaterialSampling result;

        switch (mtrl->type) {
        case aten::MaterialType::Emissive:
            result.bsdf = AT_NAME::emissive::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Diffuse:
            result.bsdf = AT_NAME::Diffuse::bsdf(mtrl);
            break;
        case aten::MaterialType::OrneNayar:
            result.bsdf = AT_NAME::OrenNayar::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Specular:
            result.bsdf = AT_NAME::specular::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Refraction:
            result.bsdf = AT_NAME::refraction::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::GGX:
            result.bsdf = AT_NAME::MicrofacetGGX::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Beckman:
            result.bsdf = AT_NAME::MicrofacetBeckman::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Velvet:
            result.bsdf = AT_NAME::MicrofacetVelvet::bsdf(mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            result.bsdf = AT_NAME::MicrofacetRefraction::bsdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::Retroreflective:
            result.bsdf = AT_NAME::Retroreflective::bsdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::CarPaint:
            result.bsdf = AT_NAME::CarPaint::bsdf(mtrl, normal, wi, wo, u, v, pre_sampled_r);
            break;
        case aten::MaterialType::Disney:
            result = AT_NAME::DisneyBRDF::bsdf(*mtrl, normal, wi, wo, u, v);
            break;
        case aten::MaterialType::ToonSpecular:
            result.bsdf = AT_NAME::ToonSpecular::ComputeBRDF(*mtrl, normal, wi, wo, u, v);
            break;
        default:
            AT_ASSERT(false);
            result.bsdf = AT_NAME::Diffuse::bsdf(mtrl);
            break;
        }

        return result;
    }

    inline AT_DEVICE_API float material::applyNormal(
        const aten::MaterialParameter* mtrl,
        const int32_t normalMapIdx,
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        float u, float v,
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
            return float(-1);
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
                // Replace to Diffuse.
                const auto& albedo = ctxt.GetMaterial(static_cast<uint32_t>(mtrl_id)).baseColor;
                dst_mtrl = aten::MaterialParameter(aten::MaterialType::Diffuse, aten::MaterialAttributeDiffuse);
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
                    dst_mtrl.albedoMap = static_cast<int32_t>(dst_mtrl.albedoMap >= 0 ? ctxt.textures[dst_mtrl.albedoMap] : -1);
                    dst_mtrl.normalMap = static_cast<int32_t>(dst_mtrl.normalMap >= 0 ? ctxt.textures[dst_mtrl.normalMap] : -1);
                    dst_mtrl.roughnessMap = static_cast<int32_t>(dst_mtrl.roughnessMap >= 0 ? ctxt.textures[dst_mtrl.roughnessMap] : -1);

                    dst_mtrl.toon.remap_texture = static_cast<int32_t>(dst_mtrl.toon.remap_texture >= 0 ? ctxt.textures[dst_mtrl.toon.remap_texture] : -1);
                }
            }
        }
        else {
            // TODO
            dst_mtrl = aten::MaterialParameter(aten::MaterialType::Diffuse, aten::MaterialAttributeDiffuse);
            dst_mtrl.baseColor = aten::vec3(1.0f);
        }

        return is_valid_mtrl;
    }

}
