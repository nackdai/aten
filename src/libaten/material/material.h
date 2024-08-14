#pragma once

#include <array>
#include <functional>
#include <string>

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"
#include "math/ray.h"
#include "misc/misc.h"

namespace AT_NAME {
    class Light;
}

namespace aten {
    class Values;
}

namespace aten
{
    struct MaterialAttribute {
        uint32_t isEmissive : 1;
        uint32_t isSingular : 1;
        uint32_t isTranslucent : 1;
        uint32_t isGlossy : 1;
    };

    AT_DEVICE_API constexpr auto MaterialAttributeMicrofacet = aten::MaterialAttribute{ false, false, false, true };
    AT_DEVICE_API constexpr auto MaterialAttributeLambert = aten::MaterialAttribute{ false, false, false, false };
    AT_DEVICE_API constexpr auto MaterialAttributeEmissive = aten::MaterialAttribute{ true,  false, false, false };
    AT_DEVICE_API constexpr auto MaterialAttributeSpecular = aten::MaterialAttribute{ false, true,  false, true };
    AT_DEVICE_API constexpr auto MaterialAttributeRefraction = aten::MaterialAttribute{ false, true,  true,  true };
    AT_DEVICE_API constexpr auto MaterialAttributeTransmission = aten::MaterialAttribute{ false, false, true,  false };

    enum class MaterialType : int32_t {
        Emissive,
        Lambert,
        OrneNayar,
        Specular,
        Refraction,
        GGX,
        Beckman,
        Velvet,
        Microfacet_Refraction,
        Retroreflective,
        CarPaint,
        Disney,

        // TODO:
        // Currently, if MaterialTypeMax is specified, it is treated as non subsurface volumetric material.
        // We may need to introduce the specific enum value.
        MaterialTypeMax,
    };

    struct StandardMaterialParameter {
        float ior;               // Index of refraction.

        float roughness;         // 表面の粗さで，ディフューズとスペキュラーレスポンスの両方を制御します.

        float shininess;
        float subsurface;        // 表面下の近似を用いてディフューズ形状を制御する.
        float metallic;          // 金属度(0 = 誘電体, 1 = 金属)。これは2つの異なるモデルの線形ブレンドです。金属モデルはディフューズコンポーネントを持たず，また色合い付けされた入射スペキュラーを持ち，基本色に等しくなります.
        float specular;          // 入射鏡面反射量。これは明示的な屈折率の代わりにあります.
        float specularTint;      // 入射スペキュラーを基本色に向かう色合いをアーティスティックな制御するための譲歩。グレージングスペキュラーはアクロマティックのままです.
        float anisotropic;       // 異方性の度合い。これはスペキュラーハイライトのアスペクト比を制御します(0 = 等方性, 1 = 最大異方性).
        float sheen;             // 追加的なグレージングコンポーネント，主に布に対して意図している.
        float sheenTint;         // 基本色に向かう光沢色合いの量.
        float clearcoat;         // 第二の特別な目的のスペキュラーローブ.
        float clearcoatGloss;    // クリアコートの光沢度を制御する(0 = “サテン”風, 1 = “グロス”風).

        AT_HOST_DEVICE_API void Init()
        {
            ior = 1.0;

            roughness = 0.5;
            shininess = 1.0;

            subsurface = 0.5;
            metallic = 0.5;
            specular = 0.5;
            specularTint = 0.5;
            anisotropic = 0.5;
            sheen = 0.5;
            sheenTint = 0.5;
            clearcoat = 0.5;
            clearcoatGloss = 0.5;
        }

        AT_HOST_DEVICE_API StandardMaterialParameter()
        {
            Init();
        }

        AT_HOST_DEVICE_API auto& operator=(const StandardMaterialParameter& rhs)
        {
            ior = rhs.ior;
            shininess = rhs.shininess;
            roughness = rhs.roughness;
            subsurface = rhs.subsurface;
            metallic = rhs.metallic;
            specular = rhs.specular;
            specularTint = rhs.specularTint;
            anisotropic = rhs.anisotropic;
            sheen = rhs.sheen;
            sheenTint = rhs.sheenTint;
            clearcoat = rhs.clearcoat;
            clearcoatGloss = rhs.clearcoatGloss;

            return *this;
        }
    };

    struct CarPaintMaterialParameter {
        aten::vec3 clearcoat_color;
        float clearcoat_ior;

        aten::vec3 flakes_color;
        float clearcoat_roughness;

        aten::vec3 diffuse_color;
        float flake_scale;

        float flake_size;
        float flake_size_variance;
        float flake_normal_orientation;
        float flake_color_multiplier;

        AT_HOST_DEVICE_API void Init()
        {
            aten::set(clearcoat_color, float(1), float(1), float(1));
            aten::set(flakes_color, float(1), float(1), float(0));
            aten::set(diffuse_color, float(1), float(0), float(1));

            clearcoat_ior = float(3.0);
            clearcoat_roughness = float(0.25);

            flake_scale = float(400.0);
            flake_size = float(0.25);
            flake_size_variance = float(0.7);
            flake_normal_orientation = float(0.5);
            flake_color_multiplier = float(1.0);
        }

        AT_HOST_DEVICE_API CarPaintMaterialParameter()
        {
            Init();
        }

        AT_HOST_DEVICE_API auto& operator=(const CarPaintMaterialParameter& rhs)
        {
            clearcoat_color = rhs.clearcoat_color;
            flakes_color = rhs.flakes_color;
            diffuse_color = rhs.diffuse_color;
            clearcoat_ior = rhs.clearcoat_ior;
            clearcoat_roughness = rhs.clearcoat_roughness;
            flake_scale = rhs.flake_scale;
            flake_size = rhs.flake_size;
            flake_size_variance = rhs.flake_size_variance;
            flake_normal_orientation = rhs.flake_normal_orientation;
            flake_color_multiplier = rhs.flake_color_multiplier;
            return *this;
        }
    };

    struct MediumParameter {
        float phase_function_g{ 0.0F };
        float sigma_a{ 0.0F };
        float sigma_s{ 0.0F };
        int32_t grid_idx{ -1 };
        float majorant{ -1.0F };
        aten::vec3 le;
    };

    struct MaterialParameter {
        aten::vec4 baseColor;   // サーフェイスカラー，通常テクスチャマップによって供給される.
        MaterialType type;

        MaterialAttribute attrib;

        uint16_t id;
        bool isIdealRefraction{ false };
        bool is_medium{ false };

        int32_t albedoMap{ -1 };
        int32_t normalMap{ -1 };
        int32_t roughnessMap{ -1 };

        union {
            StandardMaterialParameter standard;
            CarPaintMaterialParameter carpaint;
        };

        MediumParameter medium;

        AT_HOST_DEVICE_API void Init()
        {
            standard.Init();
        }

        AT_HOST_DEVICE_API MaterialParameter()
        {
            baseColor.set(float(0), float(0), float(0), float(1));

            Init();
        }

        AT_HOST_DEVICE_API MaterialParameter(MaterialType _type, const MaterialAttribute& _attrib)
            : MaterialParameter()
        {
            type = _type;
            attrib = _attrib;

            // TODO
            // multiple initialization.
            if (type == MaterialType::CarPaint) {
                carpaint.Init();
            }
        }

        AT_HOST_DEVICE_API auto& operator=(const MaterialParameter& rhs)
        {
            id = rhs.id;
            baseColor = rhs.baseColor;

            type = rhs.type;

            attrib = rhs.attrib;

            isIdealRefraction = rhs.isIdealRefraction;
            is_medium = rhs.is_medium;

            albedoMap = rhs.albedoMap;
            normalMap = rhs.normalMap;
            roughnessMap = rhs.roughnessMap;

            if (type == MaterialType::CarPaint) {
                carpaint = rhs.carpaint;
            }
            else {
                standard = rhs.standard;
            }

            medium = rhs.medium;

            return *this;
        }
    };

    class IMaterialParamEditor {
    protected:
        IMaterialParamEditor() {}
        virtual ~IMaterialParamEditor() {}

    public:
        virtual bool edit(std::string_view name, float& param, float _min = 0.0F, float _max = 1.0F) = 0;
        virtual bool edit(std::string_view name, bool& param) = 0;
        virtual bool edit(std::string_view name, vec3& param) = 0;
        virtual bool edit(std::string_view name, vec4& param) = 0;

        void editTex(std::string_view name, int32_t texid)
        {
            if (texid >= 0) {
                /*auto tex = aten::texture::GetTexture(texid);
                if (tex) {
                    edit(name, tex->name());
                }*/
            }
        }

    protected:
        virtual void edit(std::string_view name, std::string_view str) = 0;
    };

    enum class MtrlParamType {
        Vec3,
        Texture,
        Double,

        Num,
    };
}

#if defined(_WIN32) || defined(_WIN64)
#define AT_EDIT_MATERIAL_PARAM(e, param, name)    (e)->edit(#name, param.##name)
#define AT_EDIT_MATERIAL_PARAM_RANGE(e, param, name, _min, _max)    (e)->edit(#name, param.##name, _min, _max)
#define AT_EDIT_MATERIAL_PARAM_TEXTURE(e, param, name)    (e)->editTex(#name, param.##name)
#else
// TODO
// For linux, to avoid token concat error.
#define AT_EDIT_MATERIAL_PARAM(e, param, name)  false
#define AT_EDIT_MATERIAL_PARAM_RANGE(e, param, name, _min, _max)    false
#define AT_EDIT_MATERIAL_PARAM_TEXTURE(e, param, name)
#endif

namespace AT_NAME
{
    struct MaterialSampling {
        aten::vec3 dir;
        aten::vec3 bsdf;
        float pdf{ float(0) };

        AT_DEVICE_API MaterialSampling() {}
        AT_DEVICE_API MaterialSampling(const aten::vec3& d, const aten::vec3& b, float p)
            : dir(d), bsdf(b), pdf(p)
        {}
    };

    class material {
        friend class context;

        struct MaterialInfo {
            std::string name;
            std::function<material* ()> func;
        };
        static const std::array<MaterialInfo, static_cast<size_t>(aten::MaterialType::MaterialTypeMax)> mtrl_type_info;

    protected:
        material(
            aten::MaterialType type,
            const aten::MaterialAttribute& attrib)
            : m_param(type, attrib)
        {}

        material(
            aten::MaterialType type,
            const aten::MaterialAttribute& attrib,
            const aten::vec3& clr,
            float ior = 1)
            : material(type, attrib)
        {
            m_param.baseColor = clr;
            m_param.standard.ior = ior;
        }

        material(
            aten::MaterialType type,
            const aten::MaterialAttribute& attrib,
            aten::Values& val);

    public:
        static std::shared_ptr<material> CreateMaterial(
            aten::MaterialType type,
            aten::Values& value);

        static std::shared_ptr<material> CreateMaterialWithMaterialParameter(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        material() = default;
        ~material() = default;

        int32_t id() const
        {
            return m_param.id;
        }

        void setTextures(
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        const aten::MaterialParameter& param() const
        {
            return m_param;
        }
        aten::MaterialParameter& param()
        {
            return m_param;
        }

        void setName(std::string_view name)
        {
            m_name = name;
        }

        const char* name() const
        {
            return m_name.c_str();
        }

        const std::string& nameString() const
        {
            return m_name;
        }

        virtual bool edit(aten::IMaterialParamEditor* editor)
        {
            //AT_ASSERT(false);
            return false;
        }

        static const char* getMaterialTypeName(aten::MaterialType type);

        static aten::MaterialType getMaterialTypeFromMaterialTypeName(std::string_view name);

        static bool isDefaultMaterialName(const std::string& name);

        static bool isValidMaterialType(aten::MaterialType type);

        static AT_DEVICE_API bool isTranslucentByAlpha(
            const aten::MaterialParameter& param,
            float u, float v);

        static AT_DEVICE_API float getTranslucentAlpha(
            const aten::MaterialParameter& param,
            float u, float v);

        static AT_DEVICE_API float computeFresnel(
            float ni, float nt,
            const aten::vec3& wi,
            const aten::vec3& normal)
        {
            auto cosi = dot(normal, wi);
            // Potentially flip interface orientation for Fresnel equations.
            if (cosi < 0) {
                aten::swap(ni, nt);
                cosi = -cosi;
            }

            const auto nnt = ni / nt;
            const auto sini2 = float(1.0) - cosi * cosi;
            const auto sint2 = nnt * nnt * sini2;
            const auto cost = aten::sqrt(aten::cmpMax(float(0.0), float(1.0) - sint2));

            const auto rp = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
            const auto rs = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);

            const auto Rsp = (rp * rp + rs * rs) * float(0.5);
            return Rsp;
        }

        /**
         * @brief Compute schlick's fresnel.
         * @param[in] ni Index of refraction of the media on the incident side.
         * @param[in] nt Index of refraction of the media on the transmitted side.
         * @param[in] w Direction to compute fresnel.
         * @param[in] n Normal vector on surface.
         * @return fresnel.
         * @note If normal is half vector between incident vector and output vector,
         *       this API can be accept whichever incident vector or output vector as `w`.
         *       Because, dot(w, n) is computed in this API, if normal is half vector, result of dot is the same between both.
         */
        static inline AT_DEVICE_API float ComputeSchlickFresnel(
            float ni, float nt,
            const aten::vec3& w,
            const aten::vec3& n)
        {
            auto costheta = dot(w, n);

            // Potentially flip interface orientation for Fresnel equations.
            if (costheta < 0) {
                aten::swap(ni, nt);
                costheta = -costheta;
            }

            // NOTE:
            // https://qiita.com/emadurandal/items/76348ad118c36317ec5c#f%E3%83%95%E3%83%AC%E3%83%8D%E3%83%AB%E9%A0%85
            // If normal is half vector between incident vectorand output vector,
            // `w` is acceptable as whichever incident vector or output vector.
            // Because, shkick fresnel use cos_theta, if normal is half vector, cos_theta is the same between dot(V, H) and dot(L, H).

            // NOTE:
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE:
            // F_schlick(v,h) = f0 + (1 - f0)(1 - cos_theta)^5
            // f0 = pow((1 - ior) / (1 + ior), 2)
            // ior = nt / ni
            // f0 = pow((1 - ior) / (1 + ior), 2)
            //    = pow((1 - nt/ni) / (1 + nt/ni), 2)
            //    = pow((ni - nt) / (ni + nt), 2)
            auto f0 = (ni - nt) / (ni + nt);
            f0 = f0 * f0;

            return ComputeSchlickFresnelWithF0AndCosTheta(f0, costheta);
        }

        template <class F0Type>
        static inline AT_DEVICE_API F0Type ComputeSchlickFresnelWithF0AndCosTheta(
            const F0Type& f0,
            const float costheta)
        {
            const auto c = aten::saturate(1 - costheta);
            const auto c5 = c * c * c * c * c;
            const auto fresnel = f0 + (1.0F - f0) * c5;
            return fresnel;
        }

        /**
         * @brief Compute ideal reflection vector.
         * @param[in] wi Incident vector.
         * @param[in] n Normal vector on surface.
         * @return Ideal reflection vector.
         */
        static AT_DEVICE_API aten::vec3 ComputeReflectVector(
            const aten::vec3& wi,
            const aten::vec3& n)
        {
            auto wo = wi - 2 * dot(wi, n) * n;
            wo = normalize(wo);
            return wo;
        }

        /**
         * @brief Compute refract vector.
         * @param[in] ni Index of refraction of the media on the incident side.
         * @param[in] nt Index of refraction of the media on the transmitted side.
         * @param[in] wi Incident vector.
         * @param[in] n Normal vector on surface.
         * @return Refract vector.
         */
        static AT_DEVICE_API aten::vec3 ComputeRefractVector(
            float ni, float nt,
            const aten::vec3& wi,
            const aten::vec3& n)
        {
            // NOTE:
            // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
            const auto w = -wi;
            auto N = n;

            auto costheta = dot(w, n);

            if (costheta < 0.0F) {
                aten::swap(ni, nt);
                costheta = -costheta;
                N = -N;
            }

            const auto sintheta_2 = 1.0F - costheta * costheta;

            const auto ni_nt = ni / nt;
            const auto ni_nt_2 = ni_nt * ni_nt;

            auto wo = (ni_nt * costheta - aten::sqrt(1.0F - ni_nt_2 * sintheta_2)) * N - ni_nt * w;
            wo = normalize(wo);

            return wo;
        }

        static AT_DEVICE_API aten::vec4 sampleAlbedoMap(
            const aten::MaterialParameter* mtrl,
            float u, float v,
            uint32_t lod = 0);

        static AT_DEVICE_API void sampleMaterial(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            float pre_sampled_r,
#ifdef __CUDACC__
            float u, float v);
#else
            float u, float v,
            bool is_light_path = false);
#endif

        static AT_DEVICE_API float samplePDF(
            const aten::MaterialParameter* dst_mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v);

        static AT_DEVICE_API aten::vec3 sampleBSDF(
            const aten::MaterialParameter* dst_mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float u, float v,
            float pre_sampled_r);

        static AT_DEVICE_API float applyNormal(
            const aten::MaterialParameter* mtrl,
            const int32_t normalMapIdx,
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            float u, float v,
            const aten::vec3& wi,
            aten::sampler* sampler);

    protected:
        aten::MaterialParameter m_param;

        // For debug.
        std::string m_name;
    };
}
