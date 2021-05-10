#include "material/beckman.h"
#include "material/blinn.h"
#include "material/disney_brdf.h"
#include "material/emissive.h"
#include "material/ggx.h"
#include "material/lambert.h"
#include "material/lambert_refraction.h"
#include "material/material.h"
#include "material/microfacet_refraction.h"
#include "material/oren_nayar.h"
#include "material/refraction.h"
#include "material/specular.h"
#include "material/toon.h"
#include "material/velvet.h"
#include "misc/value.h"

namespace AT_NAME {
    material::material(
        aten::MaterialType type,
        const aten::MaterialAttribute& attrib,
        aten::Values& val)
        : material(type, attrib)
    {
        m_param.baseColor = val.get("baseColor", m_param.baseColor);
        m_param.ior = val.get("ior", m_param.ior);

        auto albedoMap = val.get<texture>("albedoMap");
        auto normalMap = val.get<texture>("normalMap");

        setTextures(albedoMap.get(), normalMap.get(), nullptr);
    }

    MicrofacetBeckman::MicrofacetBeckman(aten::Values& val)
        : material(aten::MaterialType::Beckman, MaterialAttributeMicrofacet, val)
    {
        m_param.roughness = val.get("roughness", m_param.roughness);
        m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    MicrofacetBlinn::MicrofacetBlinn(aten::Values& val)
        : material(aten::MaterialType::Blinn, MaterialAttributeMicrofacet, val)
    {
        m_param.shininess = val.get("shininess", m_param.shininess);
    }

    DisneyBRDF::DisneyBRDF(aten::Values& val)
        : material(aten::MaterialType::Disney, MaterialAttributeMicrofacet, val)
    {
        // TODO
        // Clamp parameters.
        m_param.subsurface = val.get("subsurface", m_param.subsurface);
        m_param.metallic = val.get("metallic", m_param.metallic);
        m_param.specular = val.get("specular", m_param.specular);
        m_param.specularTint = val.get("specularTint", m_param.specularTint);
        m_param.roughness = val.get("roughness", m_param.roughness);
        m_param.anisotropic = val.get("anisotropic", m_param.anisotropic);
        m_param.sheen = val.get("sheen", m_param.sheen);
        m_param.sheenTint = val.get("sheenTint", m_param.sheenTint);
        m_param.clearcoat = val.get("clearcoat", m_param.clearcoat);
        m_param.clearcoatGloss = val.get("clearcoatGloss", m_param.clearcoatGloss);
        m_param.roughnessMap = val.get("roughnessmap", m_param.roughnessMap);

        m_param.ior = val.get("ior", m_param.ior);
    }

    emissive::emissive(aten::Values& val)
        : material(aten::MaterialType::Emissive, MaterialAttributeEmissive, val)
    {}

    MicrofacetGGX::MicrofacetGGX(aten::Values& val)
        : material(aten::MaterialType::GGX, MaterialAttributeMicrofacet, val)
    {
        m_param.roughness = val.get("roughness", m_param.roughness);
        m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    lambert::lambert(aten::Values& val)
        : material(aten::MaterialType::Lambert, MaterialAttributeLambert, val)
    {}

    LambertRefraction::LambertRefraction(aten::Values& val)
        : material(aten::MaterialType::Lambert_Refraction, MaterialAttributeTransmission, val)
    {}

    MicrofacetRefraction::MicrofacetRefraction(aten::Values& val)
        : material(aten::MaterialType::Microfacet_Refraction, MaterialAttributeRefraction, val)
    {}

    OrenNayar::OrenNayar(aten::Values& val)
        : material(aten::MaterialType::OrneNayar, MaterialAttributeLambert, val)
    {
        m_param.roughness = val.get("roughness", m_param.roughness);
        m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);

        auto roughnessMap = val.get<texture>("roughnessmap");
        m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    refraction::refraction(aten::Values& val)
        : material(aten::MaterialType::Refraction, MaterialAttributeRefraction, val)
    {
        m_param.isIdealRefraction = val.get("isIdealRefraction", m_param.isIdealRefraction);
    }

    specular::specular(aten::Values& val)
        : material(aten::MaterialType::Specular, MaterialAttributeSpecular, val)
    {}

    toon::toon(aten::Values& val)
        : NPRMaterial(aten::MaterialType::Toon, val)
    {}

    MicrofacetVelvet::MicrofacetVelvet(aten::Values& val)
        : material(aten::MaterialType::Velvet, MaterialAttributeMicrofacet, val)
    {
        m_param.roughness = val.get("roughness", m_param.roughness);
        m_param.roughness = aten::clamp<real>(m_param.roughness, 0, 1);
    }
}
