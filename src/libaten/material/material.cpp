#include <atomic>
#include "material/material.h"
#include "light/light.h"
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
        "disney_brdf",
        "carpaint",
        "toon",
        "layer",
    };
    AT_STATICASSERT(AT_COUNTOF(g_mtrlTypeNames) == (int)aten::MaterialType::MaterialTypeMax);

    const char* material::getMaterialTypeName(aten::MaterialType type)
    {
        AT_ASSERT(static_cast<int>(type) < AT_COUNTOF(g_mtrlTypeNames));
        return g_mtrlTypeNames[type];
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

        for each (const char* mtrl_name in g_mtrlTypeNames) {
            if (lower == mtrl_name) {
                return true;
            }
        }

        return false;
    }

    bool material::isValidMaterialType(aten::MaterialType type)
    {
        return (0 <= type && type < aten::MaterialType::MaterialTypeMax);
    }

    void material::resetIdWhenAnyMaterialLeave(AT_NAME::material* mtrl)
    {
        mtrl->m_id = mtrl->m_listItem.currentIndex();
    }

    material::material(
        aten::MaterialType type, 
        const aten::MaterialAttribute& attrib)
        : m_param(type, attrib)
    {
        m_listItem.init(this, resetIdWhenAnyMaterialLeave);
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
        m_listItem.init(this, resetIdWhenAnyMaterialLeave);

        m_param.baseColor = clr;
        m_param.ior = ior;

        setTextures(albedoMap, normalMap, nullptr);
    }

    material::material(
        aten::MaterialType type, 
        const aten::MaterialAttribute& attrib, 
        aten::Values& val)
        : material(type, attrib)
    {
        m_listItem.init(this, resetIdWhenAnyMaterialLeave);

        m_param.baseColor = val.get("baseColor", m_param.baseColor);
        m_param.ior = val.get("ior", m_param.ior);
        
        auto albedoMap = (aten::texture*)val.get<void*>("albedoMap", nullptr);
        auto normalMap = (aten::texture*)val.get<void*>("normalMap", nullptr);

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

    material::~material()
    {
        m_listItem.leave();
    }

    NPRMaterial::NPRMaterial(
        aten::MaterialType type,
        const aten::vec3& e, AT_NAME::Light* light)
        : material(type, MaterialAttributeNPR, e)
    {
        setTargetLight(light);
    }

    void NPRMaterial::setTargetLight(AT_NAME::Light* light)
    {
        m_targetLight = light;
    }

    const AT_NAME::Light* NPRMaterial::getTargetLight() const
    {
        return m_targetLight;
    }

    // NOTE
    // Schlick によるフレネル反射率の近似.
    // http://yokotakenji.me/log/math/4501/
    // https://en.wikipedia.org/wiki/Schlick%27s_approximation

    // NOTE
    // フレネル反射率について.
    // http://d.hatena.ne.jp/hanecci/20130525/p3

    real schlick(
        const aten::vec3& in,
        const aten::vec3& normal,
        real ni, real nt)
    {
        // NOTE
        // Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
        // R0 = ((n1 - n2) / (n1 + n2))^2

        auto r0 = (ni - nt) / (ni + nt);
        r0 = r0 * r0;

        auto c = dot(in, normal);

        return r0 + (1 - r0) * aten::pow((1 - c), 5);
    }

    real computFresnel(
        const aten::vec3& in,
        const aten::vec3& normal,
        real ni, real nt)
    {
        real cos_i = dot(in, normal);

        bool isEnter = (cos_i > real(0));

        aten::vec3 n = normal;

        if (isEnter) {
            // レイが出ていくので、全部反対.
            auto tmp = nt;
            nt = real(1);
            ni = tmp;

            n = -n;
        }

        auto eta = ni / nt;

        auto sini2 = 1.f - cos_i * cos_i;
        auto sint2 = eta * eta * sini2;

        auto fresnel = schlick(
            in, 
            n, ni, nt);

        return fresnel;
    }

    AT_DEVICE_MTRL_API aten::vec3 material::sampleAlbedoMap(real u, real v) const
    {
        return std::move(AT_NAME::sampleTexture(m_param.albedoMap, u, v, aten::vec3(real(1))));
    }

    AT_DEVICE_MTRL_API void material::applyNormalMap(
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v) const
    {
        AT_NAME::applyNormalMap(m_param.normalMap, orgNml, newNml, u, v);
    }
}
