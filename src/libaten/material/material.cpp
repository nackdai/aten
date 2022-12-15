#include <atomic>

#include "light/light.h"
#include "material/material.h"
#include "material/sample_texture.h"

namespace AT_NAME
{
    static const char* g_mtrlTypeNames[] = {
        "emissive",
        "lambert",
        "ornenayar",
        "specular",
        "refraction",
        "blinn",
        "ggx",
        "beckman",
        "velvet",
        "lambert_refraction",
        "microfacet_refraction",
        "retroreflective",
        "carpaint",
        "disney_brdf",
        "toon",
        "layer",
    };
    AT_STATICASSERT(AT_COUNTOF(g_mtrlTypeNames) == (int)aten::MaterialType::MaterialTypeMax);

    const char* material::getMaterialTypeName(aten::MaterialType type)
    {
        AT_ASSERT(static_cast<int>(type) < AT_COUNTOF(g_mtrlTypeNames));
        return g_mtrlTypeNames[static_cast<int>(type)];
    }

    aten::MaterialType material::getMaterialTypeFromMaterialTypeName(const std::string& name)
    {
        std::string lowerName = name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        for (int i = 0; i < AT_COUNTOF(g_mtrlTypeNames); i++) {
            auto mtrlName = g_mtrlTypeNames[i];

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

        for (const char* mtrl_name : g_mtrlTypeNames) {
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

    material::material(
        aten::MaterialType type,
        const aten::MaterialAttribute& attrib)
        : m_param(type, attrib)
    {
    }

    material::material(
        aten::MaterialType type,
        const aten::MaterialAttribute& attrib,
        const aten::vec3& clr,
        real ior/*= 1*/,
        aten::texture* albedoMap/*= nullptr*/,
        aten::texture* normalMap/*= nullptr*/)
        : material(type, attrib)
    {
        m_param.baseColor = clr;
        m_param.standard.ior = ior;

        setTextures(albedoMap, normalMap, nullptr);
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

    NPRMaterial::NPRMaterial(
        aten::MaterialType type,
        const aten::vec3& e,
        const std::shared_ptr<AT_NAME::Light>& light)
        : material(type, MaterialAttributeNPR, e)
    {
        setTargetLight(light);
    }

    void NPRMaterial::setTargetLight(const std::shared_ptr<AT_NAME::Light>& light)
    {
        m_targetLight = light;
    }

    std::shared_ptr<const AT_NAME::Light> NPRMaterial::getTargetLight() const
    {
        return m_targetLight;
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
