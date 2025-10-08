#include <atomic>

#include "light/light.h"
#include "misc/value.h"
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
#include "material/toon.h"
#include "material/car_paint.h"

#include "volume/medium.h"

namespace AT_NAME
{
    const std::array<std::string, static_cast<size_t>(aten::MaterialType::MaterialTypeMax)> material::mtrl_type_info = { {
        "emissive",
        "Diffuse",
        "ornenayar",
        "specular",
        "refraction",
        "ggx",
        "beckman",
        "velvet"
        "microfacet_refraction",
        "retroreflective",
        "carpaint",
        "disney_brdf",
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

        param.is_ideal_refraction = value.get("is_ideal_refraction", param.is_ideal_refraction);

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
            mtrl = new AT_NAME::emissive(param);
            break;
        case aten::MaterialType::Diffuse:
            mtrl = new AT_NAME::Diffuse(param, albedoMap, normalMap);
            break;
        case aten::MaterialType::OrneNayar:
            mtrl = new AT_NAME::OrenNayar(param, albedoMap, normalMap);
            break;
        case aten::MaterialType::Specular:
            mtrl = new AT_NAME::specular(param, albedoMap, normalMap);
            break;
        case aten::MaterialType::Refraction:
            mtrl = new AT_NAME::refraction(param, normalMap);
            break;
        case aten::MaterialType::GGX:
            mtrl = new AT_NAME::MicrofacetGGX(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Beckman:
            mtrl = new AT_NAME::MicrofacetBeckman(param, albedoMap, normalMap, roughnessMap);
            break;
        case aten::MaterialType::Velvet:
            mtrl = new AT_NAME::MicrofacetVelvet(param, albedoMap, normalMap);
            break;
        case aten::MaterialType::Microfacet_Refraction:
            mtrl = new AT_NAME::MicrofacetRefraction(param, albedoMap, normalMap, roughnessMap);
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
        case aten::MaterialType::Toon:
            mtrl = new AT_NAME::Toon(param, albedoMap, normalMap);
            break;
        default:
            mtrl = new AT_NAME::material();
            mtrl->param_ = param;
            break;
        }

        std::shared_ptr<material> ret(mtrl);

        return ret;
    }

    const char* material::GetMaterialTypeName(aten::MaterialType type)
    {
        return mtrl_type_info[static_cast<size_t>(type)].c_str();
    }

    aten::MaterialType material::GetMaterialTypeFromMaterialTypeName(std::string_view name)
    {
        std::string lowerName{ name };
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        for (size_t i = 0; i < mtrl_type_info.size(); i++) {
            const auto& mtrlName = mtrl_type_info[i];

            if (lowerName == mtrlName) {
                return static_cast<aten::MaterialType>(i);
            }
        }

        AT_ASSERT(false);
        return aten::MaterialType::MaterialTypeMax;
    }

    bool material::IsDefaultMaterialName(const std::string& name)
    {
        auto lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        for (const auto& info : mtrl_type_info) {
            const auto& mtrl_name = info;
            if (lower == mtrl_name) {
                return true;
            }
        }

        return false;
    }

    bool material::IsValidMaterialType(aten::MaterialType type)
    {
        return (0 <= static_cast<int32_t>(type)
            && static_cast<int32_t>(type) < static_cast<int32_t>(aten::MaterialType::MaterialTypeMax));
    }

    void material::SetTextures(
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        param_.albedoMap = albedoMap ? albedoMap->id() : -1;
        param_.normalMap = normalMap ? normalMap->id() : -1;
        param_.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
    }

    AT_DEVICE_API bool material::IsTranslucentByAlpha(
        const aten::MaterialParameter& param,
        float u, float v)
    {
        auto alpha = GetTranslucentAlpha(param, u, v);
        return alpha < float(1);
    }

    AT_DEVICE_API float material::GetTranslucentAlpha(
        const aten::MaterialParameter& param,
        float u, float v)
    {
        auto albedo{ AT_NAME::sampleTexture(param.albedoMap, u, v, aten::vec4(float(1))) };
        albedo *= param.baseColor;
        return albedo.a;
    }

    aten::MaterialParameter material::CreateMaterialMediumParameter(
        const float g,
        const float sigma_a,
        const float sigma_s,
        const aten::vec3& le)
    {
        aten::MaterialParameter mtrl;

        mtrl.type = aten::MaterialType::Volume;

        mtrl.is_medium = true;
        mtrl.medium.phase_function_g = g;
        mtrl.medium.sigma_a = sigma_a;
        mtrl.medium.sigma_s = sigma_s;
        mtrl.medium.le = le;

        mtrl.attrib.is_emissive = false;
        mtrl.attrib.is_singular = false;
        mtrl.attrib.is_glossy = false;
        mtrl.attrib.is_translucent = false;

        return mtrl;
    }
}
