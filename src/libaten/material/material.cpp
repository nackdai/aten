#include <atomic>

#include "light/light.h"
#include "misc/value.h"
#include "material/material.h"
#include "material/sample_texture.h"
#include "material/emissive.h"
#include "material/lambert.h"
#include "material/oren_nayar.h"
#include "material/specular.h"
#include "material/refraction.h"
#include "material/ggx.h"
#include "material/beckman.h"
#include "material/velvet.h"
#include "material/microfacet_refraction.h"
#include "material/disney_brdf.h"
#include "material/retroreflective.h"
#include "material/car_paint.h"

namespace AT_NAME
{
    const std::array<material::MaterialInfo, static_cast<size_t>(aten::MaterialType::MaterialTypeMax)> material::mtrl_type_info = { {
        {"emissive", []() { return new emissive(); }},
        {"lambert", []() { return new lambert(); }},
        {"ornenayar", []() { return new OrenNayar(); }},
        {"specular", []() { return new specular(); }},
        {"refraction", []() { return new refraction(); }},
        {"ggx", []() { return new MicrofacetGGX(); }},
        {"beckman", []() { return new MicrofacetBeckman(); }},
        {"velvet", []() { return new MicrofacetVelvet(); }},
        {"microfacet_refraction", []() { return new MicrofacetRefraction(); }},
        {"retroreflective", []() { return new Retroreflective(); }},
        {"carpaint", []() { return new CarPaint(); }},
        {"disney_brdf", []() { return new DisneyBRDF(); }},
    } };

    std::shared_ptr<material> material::CreateMaterial(
        aten::MaterialType type,
        aten::Values& value)
    {
        auto albedoMap = value.get<aten::texture>("albedoMap");
        auto normalMap = value.get<aten::texture>("normalMap");
        auto roughnessMap = value.get<aten::texture>("roughnessMap");

        aten::MaterialParameter param;
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

        return CreateMaterialWithMaterialParameter(
            param, albedoMap.get(), normalMap.get(), roughnessMap.get());
    }

    std::shared_ptr<material> material::CreateMaterialWithMaterialParameter(
        const aten::MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        AT_NAME::material* mtrl = nullptr;

        switch (param.type) {
        case aten::MaterialType::Emissive:
            mtrl = new AT_NAME::emissive(param.baseColor);
            break;
        case aten::MaterialType::Lambert:
            mtrl = new AT_NAME::lambert(param.baseColor, albedoMap, normalMap);
            break;
        case aten::MaterialType::OrneNayar:
            mtrl = new AT_NAME::OrenNayar(param.baseColor, param.standard.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Specular:
            mtrl = new AT_NAME::specular(param.baseColor, param.standard.ior, albedoMap, normalMap);
            break;
        case aten::MaterialType::Refraction:
            mtrl = new AT_NAME::refraction(param.baseColor, param.standard.ior, param.isIdealRefraction, normalMap);
            break;
        case aten::MaterialType::GGX:
            mtrl = new AT_NAME::MicrofacetGGX(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Beckman:
            mtrl = new AT_NAME::MicrofacetBeckman(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Velvet:
            mtrl = new AT_NAME::MicrofacetVelvet(param.baseColor, param.standard.roughness, albedoMap, normalMap);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            mtrl = new AT_NAME::MicrofacetRefraction(param.baseColor, param.standard.roughness, param.standard.ior, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Retroreflective:
            mtrl = new AT_NAME::Retroreflective(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::CarPaint:
            mtrl = new AT_NAME::CarPaint(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Disney:
            mtrl = new AT_NAME::DisneyBRDF(param, albedoMap, normalMap, roughnessMap);
            break;
        default:
            mtrl = new AT_NAME::material();
            mtrl->m_param = param;
            break;
        }

        std::shared_ptr<material> ret(mtrl);

        return ret;
    }

    const char* material::getMaterialTypeName(aten::MaterialType type)
    {
        return mtrl_type_info[static_cast<size_t>(type)].name.c_str();
    }

    aten::MaterialType material::getMaterialTypeFromMaterialTypeName(std::string_view name)
    {
        std::string lowerName{ name };
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        for (size_t i = 0; i < mtrl_type_info.size(); i++) {
            const auto& mtrlName = mtrl_type_info[i].name;

            if (lowerName == mtrlName) {
                return static_cast<aten::MaterialType>(i);
            }
        }

        AT_ASSERT(false);
        return aten::MaterialType::MaterialTypeMax;
    }

    bool material::isDefaultMaterialName(const std::string& name)
    {
        auto lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        for (const auto& info : mtrl_type_info) {
            const auto& mtrl_name = info.name;
            if (lower == mtrl_name) {
                return true;
            }
        }

        return false;
    }

    bool material::isValidMaterialType(aten::MaterialType type)
    {
        return (0 <= static_cast<int32_t>(type)
            && static_cast<int32_t>(type) < static_cast<int32_t>(aten::MaterialType::MaterialTypeMax));
    }

    void material::setTextures(
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        m_param.albedoMap = albedoMap ? albedoMap->id() : -1;
        m_param.normalMap = normalMap ? normalMap->id() : -1;
        m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    AT_DEVICE_API bool material::isTranslucentByAlpha(
        const aten::MaterialParameter& param,
        float u, float v)
    {
        auto alpha = getTranslucentAlpha(param, u, v);
        return alpha < float(1);
    }

    AT_DEVICE_API float material::getTranslucentAlpha(
        const aten::MaterialParameter& param,
        float u, float v)
    {
        auto albedo{ AT_NAME::sampleTexture(param.albedoMap, u, v, aten::vec4(float(1))) };
        albedo *= param.baseColor;
        return albedo.a;
    }
}
