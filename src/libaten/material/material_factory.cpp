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
#include "material/retroreflective.h"
#include "material/car_paint.h"
#include "misc/value.h"

namespace aten
{
    std::shared_ptr<material> MaterialFactory::createMaterial(
        MaterialType type,
        Values& value)
    {
        auto albedoMap = value.get<texture>("albedoMap");
        auto normalMap = value.get<texture>("normalMap");
        auto roughnessMap = value.get<texture>("roughnessMap");

        MaterialParameter param;
        {
            param.type = type;
            param.baseColor = value.get("baseColor", param.baseColor);
            param.standard.ior = value.get("ior", param.standard.ior);
        }

        param.standard.roughness = value.get("roughness", param.standard.roughness);
        param.standard.shininess = value.get("shininess", param.standard.shininess);
        param.standard.subsurface = value.get("subsurface", param.standard.subsurface);
        param.standard.metallic = value.get("metallic", param.standard.metallic);
        param.standard.specular = value.get("specular", param.standard.specular);
        param.standard.specularTint = value.get("specularTint", param.standard.specularTint);
        param.standard.anisotropic = value.get("anisotropic", param.standard.anisotropic);
        param.standard.sheen = value.get("sheen", param.standard.sheen);
        param.standard.sheenTint = value.get("sheenTint", param.standard.sheenTint);
        param.standard.clearcoat = value.get("clearcoat", param.standard.clearcoat);
        param.standard.clearcoatGloss = value.get("clearcoatGloss", param.standard.clearcoatGloss);

        param.isIdealRefraction = value.get("isIdealRefraction", param.isIdealRefraction);

        return createMaterialWithMaterialParameter(
            param, albedoMap.get(), normalMap.get(), roughnessMap.get());
    }

    std::shared_ptr<material> MaterialFactory::createMaterialWithDefaultValue(MaterialType type)
    {
        AT_ASSERT(material::isValidMaterialType(type));

        // NOTE
        // ctor is private. So, create an instance with new.
        // And then, pass it to shared_ptr.
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
            []() { return new Retroreflective(); },      // retroreflective
            []() { return new CarPaint(); },             // carpaint
            []() { return new DisneyBRDF(); },           // disney_brdf
            []() { return nullptr; },                    // layer
        };
        AT_STATICASSERT(AT_COUNTOF(funcs) == (int)aten::MaterialType::MaterialTypeMax);

        std::shared_ptr<material> ret(funcs[static_cast<int>(type)]());
        return ret;
    }

    std::shared_ptr<material> MaterialFactory::createMaterialWithMaterialParameter(
        const MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        aten::material* mtrl = nullptr;

        switch (param.type) {
        case aten::MaterialType::Emissive:
            mtrl = new aten::emissive(param.baseColor);
            break;
        case aten::MaterialType::Lambert:
            mtrl = new aten::lambert(param.baseColor, albedoMap, normalMap);
            break;
        case aten::MaterialType::OrneNayar:
            mtrl = new aten::OrenNayar(param.baseColor, param.standard.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Specular:
            mtrl = new aten::specular(param.baseColor, param.standard.ior, albedoMap, normalMap);
            break;
        case aten::MaterialType::Refraction:
            mtrl = new aten::refraction(param.baseColor, param.standard.ior, param.isIdealRefraction, normalMap);
            break;
        case aten::MaterialType::Blinn:
            mtrl = new aten::MicrofacetBlinn(param.baseColor, param.standard.shininess, param.standard.ior, albedoMap, normalMap);
            break;
        case aten::MaterialType::GGX:
            mtrl = new aten::MicrofacetGGX(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Beckman:
            mtrl = new aten::MicrofacetBeckman(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Velvet:
            mtrl = new aten::MicrofacetVelvet(param.baseColor, param.standard.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Lambert_Refraction:
            mtrl = new aten::LambertRefraction(param.baseColor, param.standard.ior, normalMap);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            mtrl = new aten::MicrofacetRefraction(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Retroreflective:
            mtrl = new aten::Retroreflective(albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::CarPaint:
            mtrl = new aten::CarPaint(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Disney:
            mtrl = new aten::DisneyBRDF(param, albedoMap, normalMap, roughnessMap);
            break;
        default:
            AT_ASSERT(false);
            AT_PRINTF("No material type [%s(%d)]\n", __FILE__, __LINE__);
            mtrl = new aten::lambert();
            break;
        }

        std::shared_ptr<material> ret(mtrl);

        return ret;
    }
}
