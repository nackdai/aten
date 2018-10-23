#include "material/material_factory.h"
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
#include "material/carpaint.h"

namespace aten
{
    material* MaterialFactory::createMaterial(
        MaterialType type,
        Values& value)
    {
        aten::texture* albedoMap = (aten::texture*)value.get<void*>("albedoMap", nullptr);
        aten::texture* normalMap = (aten::texture*)value.get<void*>("normalMap", nullptr);
        aten::texture* roughnessMap = (aten::texture*)value.get<void*>("roughnessMap", nullptr);

        MaterialParameter param;
        {
            param.baseColor = value.get("baseColor", param.baseColor);
            param.ior = value.get("ior", param.ior);
        }

        if (type == MaterialType::CarPaint) {
            param.carpaint.clearcoatRoughness = value.get("clearcoatRoughness", param.carpaint.clearcoatRoughness);
            param.carpaint.flakeLayerRoughness = value.get("flakeLayerRoughness", param.carpaint.flakeLayerRoughness);
            param.carpaint.flake_scale = value.get("flake_scale", param.carpaint.flake_scale);
            param.carpaint.flake_size = value.get("flake_size", param.carpaint.flake_size);
            param.carpaint.flake_size_variance = value.get("flake_size_variance", param.carpaint.flake_size_variance);
            param.carpaint.flake_normal_orientation = value.get("flake_normal_orientation", param.carpaint.flake_normal_orientation);
            param.carpaint.flake_reflection = value.get("flake_reflection", param.carpaint.flake_reflection);
            param.carpaint.flake_transmittance = value.get("flake_transmittance", param.carpaint.flake_transmittance);
            param.carpaint.glitterColor = value.get("glitterColor", param.carpaint.glitterColor);
            param.carpaint.flakeColor = value.get("clearcoat", param.carpaint.flakeColor);
            param.carpaint.flake_intensity = value.get("clearcoatGloss", param.carpaint.flake_intensity);
        }
        else {
            param.roughness = value.get("roughness", param.roughness);
            param.shininess = value.get("shininess", param.shininess);
            param.subsurface = value.get("subsurface", param.subsurface);
            param.metallic = value.get("metallic", param.metallic);
            param.specular = value.get("specular", param.specular);
            param.specularTint = value.get("specularTint", param.specularTint);
            param.anisotropic = value.get("anisotropic", param.anisotropic);
            param.sheen = value.get("sheen", param.sheen);
            param.sheenTint = value.get("sheenTint", param.sheenTint);
            param.clearcoat = value.get("clearcoat", param.clearcoat);
            param.clearcoatGloss = value.get("clearcoatGloss", param.clearcoatGloss);

            param.isIdealRefraction = value.get("isIdealRefraction", param.isIdealRefraction);
        }

        return createMaterialWithMaterialParameter(type, param, albedoMap, normalMap, roughnessMap);
    }

    material* MaterialFactory::createMaterialWithDefaultValue(MaterialType type)
    {
        AT_ASSERT(material::isValidMaterialType(type));

        std::function<material*()> funcs[] = {
            []() { return new emissive(); },             // emissive
            []() { return new lambert(); },              // lambert
            []() { return new OrenNayar(); },            // oren_nayar
            []() { return new specular(); },             // specular
            []() { return new refraction(); },           // refraction
            []() { return new MicrofacetBlinn(); },      // blinn
            []() { return new MicrofacetGGX(); },        // ggx
            []() { return new MicrofacetBeckman(); },    // beckman
            []() { return new MicrofacetVelvet(); },     // velvet
            []() { return new LambertRefraction(); },    // lambert_rafraction
            []() { return new MicrofacetRefraction(); }, // microfacet_rafraction
            []() { return new DisneyBRDF(); },           // disney_brdf
            []() { return new CarPaintBRDF(); },         // carpaint
            []() { return nullptr; },                    // toon
            []() { return nullptr; },                    // layer
        };
        AT_STATICASSERT(AT_COUNTOF(funcs) == (int)aten::MaterialType::MaterialTypeMax);

        return funcs[type]();
    }

    material* MaterialFactory::createMaterialWithMaterialParameter(
        MaterialType type,
        const MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        aten::material* mtrl = nullptr;

        switch (type) {
        case aten::MaterialType::Emissive:
            mtrl = new aten::emissive(param.baseColor);
            break;
        case aten::MaterialType::Lambert:
            mtrl = new aten::lambert(param.baseColor, albedoMap, normalMap);
            break;
        case aten::MaterialType::OrneNayar:
            mtrl = new aten::OrenNayar(param.baseColor, param.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Specular:
            mtrl = new aten::specular(param.baseColor, param.ior, albedoMap, normalMap);
            break;
        case aten::MaterialType::Refraction:
            mtrl = new aten::refraction(param.baseColor, param.ior, param.isIdealRefraction, normalMap);
            break;
        case aten::MaterialType::Blinn:
            mtrl = new aten::MicrofacetBlinn(param.baseColor, param.shininess, param.ior, albedoMap, normalMap);
            break;
        case aten::MaterialType::GGX:
            mtrl = new aten::MicrofacetGGX(param.baseColor, param.roughness, param.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Beckman:
            mtrl = new aten::MicrofacetBeckman(param.baseColor, param.roughness, param.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Velvet:
            mtrl = new aten::MicrofacetVelvet(param.baseColor, param.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Lambert_Refraction:
            mtrl = new aten::LambertRefraction(param.baseColor, param.ior, normalMap);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            mtrl = new aten::MicrofacetRefraction(param.baseColor, param.roughness, param.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Disney:
            mtrl = new aten::DisneyBRDF(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::CarPaint:
            mtrl = new aten::CarPaintBRDF(param, roughnessMap);
            break;
        default:
            AT_ASSERT(false);
            mtrl = new aten::lambert();
            break;
        }

        return mtrl;
    }
}
