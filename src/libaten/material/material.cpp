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
    const std::array<material::MaterialInfo, static_cast<size_t>(aten::MaterialType::MaterialTypeMax)> material::mtrl_type_info = { {
        {"emissive", []() { return new emissive(); }},
        {"lambert", []() { return new lambert(); }},
        {"ornenayar", []() { return new OrenNayar(); }},
        {"specular", []() { return new specular(); }},
        {"refraction", []() { return new refraction(); }},
        {"blinn", []() { return new MicrofacetBlinn(); }},
        {"ggx", []() { return new MicrofacetGGX(); }},
        {"beckman", []() { return new MicrofacetBeckman(); }},
        {"velvet", []() { return new MicrofacetVelvet(); }},
        {"lambert_refraction", []() { return new LambertRefraction(); }},
        {"microfacet_refraction", []() { return new MicrofacetRefraction(); }},
        {"retroreflective", []() { return new Retroreflective(); }},
        {"carpaint", []() { return new CarPaint(); }},
        {"disney_brdf", []() { return new DisneyBRDF(); }},
    } };

    std::shared_ptr<material> material::createMaterial(
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

    std::shared_ptr<material> material::createMaterialWithDefaultValue(MaterialType type)
    {
        AT_ASSERT(material::isValidMaterialType(type));
        std::shared_ptr<material> ret(mtrl_type_info[static_cast<size_t>(type)].func());
        return ret;
    }

    std::shared_ptr<material> material::createMaterialWithMaterialParameter(
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
        return (0 <= static_cast<int>(type)
            && static_cast<int>(type) < static_cast<int>(aten::MaterialType::MaterialTypeMax));
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

    AT_DEVICE_MTRL_API aten::vec4 material::sampleAlbedoMap(real u, real v) const
    {
        return AT_NAME::sampleTexture(m_param.albedoMap, u, v, aten::vec4(real(1)));
    }

    AT_DEVICE_MTRL_API real material::applyNormalMap(
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v,
        const aten::vec3& wi,
        aten::sampler* sampler) const
    {
        AT_NAME::applyNormalMap(m_param.normalMap, orgNml, newNml, u, v);
        return real(-1);
    }

    AT_DEVICE_MTRL_API bool material::isTranslucentByAlpha(
        const aten::MaterialParameter& param,
        real u, real v)
    {
        auto alpha = getTranslucentAlpha(param, u, v);
        return alpha < real(1);
    }

    AT_DEVICE_MTRL_API real material::getTranslucentAlpha(
        const aten::MaterialParameter& param,
        real u, real v)
    {
        auto albedo{ AT_NAME::sampleTexture(param.albedoMap, u, v, aten::vec4(real(1))) };
        albedo *= param.baseColor;
        return albedo.a;
    }
}
