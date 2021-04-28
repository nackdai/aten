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

        for (const char* mtrl_name : g_mtrlTypeNames) {
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
        m_param.ior = ior;

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

    // NOTE
    // Schlickによるフレネル反射率の近似.
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

    AT_DEVICE_MTRL_API aten::vec4 material::sampleMultipliedAlbedo(real u, real v) const
    {
        return sampleAlbedoMap(u, v) * m_param.baseColor;
    }

    AT_DEVICE_MTRL_API bool material::sampleAlphaBlend(
        AlphaBlendedMaterialSampling& result,
        real accumulatedAlpha,
        const aten::vec4& multipliedAlbedo,
        const aten::ray& ray,
        const aten::vec3& point,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v)
    {
        auto alpha = aten::clamp(multipliedAlbedo.a, real(0), real(1));

        auto alphaR = sampler->nextSample();

        if (alphaR > alpha) {
            result.pdf = real(1);

            auto nmlForAlphaBlend = dot(ray.dir, orgnormal) < real(0)
                ? -orgnormal
                : orgnormal;

            result.ray = aten::ray(point, ray.dir, nmlForAlphaBlend);

            result.bsdf = accumulatedAlpha * alpha * static_cast<aten::vec3>(multipliedAlbedo);
            result.alpha = alpha;

            return true;
        }

        return false;
    }

    AT_DEVICE_MTRL_API aten::vec4 material::sampleAlbedoMap(real u, real v) const
    {
        return AT_NAME::sampleTexture(m_param.albedoMap, u, v, aten::vec4(real(1)));
    }

    AT_DEVICE_MTRL_API void material::applyNormalMap(
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v) const
    {
        AT_NAME::applyNormalMap(m_param.normalMap, orgNml, newNml, u, v);
    }
}
