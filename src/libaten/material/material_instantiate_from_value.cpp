#include "material/beckman.h"
#include "material/disney_brdf.h"
#include "material/emissive.h"
#include "material/ggx.h"
#include "material/diffuse.h"
#include "material/material.h"
#include "material/microfacet_refraction.h"
#include "material/oren_nayar.h"
#include "material/refraction.h"
#include "material/specular.h"
#include "material/velvet.h"
#include "misc/value.h"

namespace AT_NAME {
    material::material(
        aten::MaterialType type,
        const aten::MaterialAttribute& attrib,
        aten::Values& val)
        : material(type, attrib)
    {
        param_.baseColor = val.get("baseColor", param_.baseColor);
        param_.standard.ior = val.get("ior", param_.standard.ior);

        auto albedoMap = val.get<texture>("albedoMap");
        auto normalMap = val.get<texture>("normalMap");

        SetTextures(albedoMap.get(), normalMap.get(), nullptr);
    }

    MicrofacetBeckman::MicrofacetBeckman(aten::Values& val)
        : material(aten::MaterialType::Beckman, MaterialAttributeMicrofacet, val)
    {
        param_.standard.roughness = val.get("roughness", param_.standard.roughness);
        param_.standard.roughness = aten::clamp<float>(param_.standard.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        param_.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    DisneyBRDF::DisneyBRDF(aten::Values& val)
        : material(aten::MaterialType::Disney, MaterialAttributeMicrofacet, val)
    {
        // TODO
        // Clamp parameters.
        param_.standard.subsurface = val.get("subsurface", param_.standard.subsurface);
        param_.standard.metallic = val.get("metallic", param_.standard.metallic);
        param_.standard.specular = val.get("specular", param_.standard.specular);
        param_.standard.specularTint = val.get("specularTint", param_.standard.specularTint);
        param_.standard.roughness = val.get("roughness", param_.standard.roughness);
        param_.standard.anisotropic = val.get("anisotropic", param_.standard.anisotropic);
        param_.standard.sheen = val.get("sheen", param_.standard.sheen);
        param_.standard.sheenTint = val.get("sheenTint", param_.standard.sheenTint);
        param_.standard.clearcoat = val.get("clearcoat", param_.standard.clearcoat);
        param_.standard.clearcoatGloss = val.get("clearcoatGloss", param_.standard.clearcoatGloss);
        param_.roughnessMap = val.get("roughnessmap", param_.roughnessMap);

        param_.standard.ior = val.get("ior", param_.standard.ior);
    }

    emissive::emissive(aten::Values& val)
        : material(aten::MaterialType::Emissive, MaterialAttributeEmissive, val)
    {}

    MicrofacetGGX::MicrofacetGGX(aten::Values& val)
        : material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, val)
    {
        param_.standard.roughness = val.get("roughness", param_.standard.roughness);
        param_.standard.roughness = aten::clamp<float>(param_.standard.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        param_.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    Diffuse::Diffuse(aten::Values& val)
        : material(aten::MaterialType::Diffuse, MaterialAttributeDiffuse, val)
    {}

    MicrofacetRefraction::MicrofacetRefraction(aten::Values& val)
        : material(aten::MaterialType::Microfacet_Refraction, MaterialAttributeRefraction, val)
    {}

    OrenNayar::OrenNayar(aten::Values& val)
        : material(aten::MaterialType::OrneNayar, MaterialAttributeDiffuse, val)
    {
        param_.standard.roughness = val.get("roughness", param_.standard.roughness);
        param_.standard.roughness = aten::clamp<float>(param_.standard.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        param_.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    refraction::refraction(aten::Values& val)
        : material(aten::MaterialType::Refraction, MaterialAttributeRefraction, val)
    {
        param_.is_ideal_refraction = val.get("is_ideal_refraction", param_.is_ideal_refraction);
    }

    specular::specular(aten::Values& val)
        : material(aten::MaterialType::Specular, MaterialAttributeSpecular, val)
    {}

    MicrofacetVelvet::MicrofacetVelvet(aten::Values& val)
        : material(aten::MaterialType::Velvet, MaterialAttributeMicrofacet, val)
    {
        param_.standard.roughness = val.get("roughness", param_.standard.roughness);
        param_.standard.roughness = aten::clamp<float>(param_.standard.roughness, 0, 1);
    }
}
