#pragma once

#include <array>
#include <functional>
#include <string>

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"
#include "math/ray.h"

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

    AT_DEVICE_API constexpr auto MaterialAttributeMicrofacet = aten::MaterialAttribute{false, false, false, true};
    AT_DEVICE_API constexpr auto MaterialAttributeLambert = aten::MaterialAttribute{false, false, false, false};
    AT_DEVICE_API constexpr auto MaterialAttributeEmissive = aten::MaterialAttribute{true,  false, false, false};
    AT_DEVICE_API constexpr auto MaterialAttributeSpecular = aten::MaterialAttribute{false, true,  false, true};
    AT_DEVICE_API constexpr auto MaterialAttributeRefraction = aten::MaterialAttribute{false, true,  true,  true};
    AT_DEVICE_API constexpr auto MaterialAttributeTransmission = aten::MaterialAttribute{false, false, true,  false};

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

        MaterialTypeMax,
    };

    struct StandardMaterialParameter {
        // NOTE
        // https://www.cs.uaf.edu/2012/spring/cs481/section/0/lecture/02_14_refraction.html
        // - Index Of Refraction
        //     Water's index of refraction is a mild 1.3; diamond's is a high 2.4.
        // - eta
        //   屈折率の比.
        //   ex) eta = 1.0 / 1.4  : air/glass's index of refraction.
        real ior;               // 屈折率.

        real roughness;         // 表面の粗さで，ディフューズとスペキュラーレスポンスの両方を制御します.

        real shininess;
        real subsurface;        // 表面下の近似を用いてディフューズ形状を制御する.
        real metallic;          // 金属度(0 = 誘電体, 1 = 金属)。これは2つの異なるモデルの線形ブレンドです。金属モデルはディフューズコンポーネントを持たず，また色合い付けされた入射スペキュラーを持ち，基本色に等しくなります.
        real specular;          // 入射鏡面反射量。これは明示的な屈折率の代わりにあります.
        real specularTint;      // 入射スペキュラーを基本色に向かう色合いをアーティスティックな制御するための譲歩。グレージングスペキュラーはアクロマティックのままです.
        real anisotropic;       // 異方性の度合い。これはスペキュラーハイライトのアスペクト比を制御します(0 = 等方性, 1 = 最大異方性).
        real sheen;             // 追加的なグレージングコンポーネント，主に布に対して意図している.
        real sheenTint;         // 基本色に向かう光沢色合いの量.
        real clearcoat;         // 第二の特別な目的のスペキュラーローブ.
        real clearcoatGloss;    // クリアコートの光沢度を制御する(0 = “サテン”風, 1 = “グロス”風).

        AT_HOST_DEVICE_API void Init()
        {
            ior = 1.0;

            roughness = 0.5;
            shininess = 1.0;

            subsurface = 0.0;
            metallic = 0.0;
            specular = 0.5;
            specularTint = 0.0;
            anisotropic = 0.0;
            sheen = 0.0;
            sheenTint = 0.5;
            clearcoat = 0.0;
            clearcoatGloss = 1.0;
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

    struct RetroreflectiveMaterialParameter {
        aten::vec3 clearcoat_color;
        real clearcoat_ior;

        aten::vec3 retrorelective_color;
        real clearcoat_roughness;

        aten::vec3 diffuse_color;
        real retrorelective_ior;

        real retrorelective_roughness;
        real padding[3];

        AT_HOST_DEVICE_API void Init()
        {
            aten::set(clearcoat_color, real(1), real(1), real(1));
            aten::set(retrorelective_color, real(1), real(1), real(1));
            aten::set(diffuse_color, real(1), real(0), real(0));

            clearcoat_ior = real(3.0);
            clearcoat_roughness = real(0.25);

            retrorelective_ior = real(3.0);
            retrorelective_roughness = real(0.26);
        }

        AT_HOST_DEVICE_API RetroreflectiveMaterialParameter()
        {
            Init();
        }

        AT_HOST_DEVICE_API auto& operator=(const RetroreflectiveMaterialParameter& rhs)
        {
            clearcoat_color = rhs.clearcoat_color;
            retrorelective_color = rhs.retrorelective_color;
            diffuse_color = rhs.diffuse_color;

            clearcoat_ior = rhs.clearcoat_ior;
            clearcoat_roughness = rhs.clearcoat_roughness;

            retrorelective_ior = rhs.retrorelective_ior;
            retrorelective_roughness = rhs.retrorelective_roughness;

            return *this;
        }
    };

    struct CarPaintMaterialParameter {
        aten::vec3 clearcoat_color;
        real clearcoat_ior;

        aten::vec3 flakes_color;
        real clearcoat_roughness;

        aten::vec3 diffuse_color;
        real flake_scale;

        real flake_size;
        real flake_size_variance;
        real flake_normal_orientation;
        real flake_color_multiplier;

        AT_HOST_DEVICE_API void Init()
        {
            aten::set(clearcoat_color, real(1), real(1), real(1));
            aten::set(flakes_color, real(1), real(1), real(0));
            aten::set(diffuse_color, real(1), real(0), real(1));

            clearcoat_ior = real(3.0);
            clearcoat_roughness = real(0.25);

            flake_scale = real(400.0);
            flake_size = real(0.25);
            flake_size_variance = real(0.7);
            flake_normal_orientation = real(0.5);
            flake_color_multiplier = real(1.0);
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

    struct MaterialParameter {
        aten::vec4 baseColor;   // サーフェイスカラー，通常テクスチャマップによって供給される.
        MaterialType type;

        MaterialAttribute attrib;

        uint16_t id;
        bool isIdealRefraction{ false };
        uint8_t padding{ 0 };

        struct {
            int32_t albedoMap;
            int32_t normalMap;
            int32_t roughnessMap;
        };

        union {
            StandardMaterialParameter standard;
            CarPaintMaterialParameter carpaint;
            RetroreflectiveMaterialParameter retrorelective;
        };

        AT_HOST_DEVICE_API void Init()
        {
            standard.Init();
        }

        AT_HOST_DEVICE_API MaterialParameter()
        {
            baseColor.set(real(0), real(0), real(0), real(1));
            isIdealRefraction = false;
            albedoMap = -1;
            normalMap = -1;
            roughnessMap = -1;

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
            else if (type == MaterialType::Retroreflective) {
                retrorelective.Init();
            }
        }

        AT_HOST_DEVICE_API auto& operator=(const MaterialParameter& rhs)
        {
            baseColor = rhs.baseColor;

            type = rhs.type;

            attrib = rhs.attrib;

            isIdealRefraction = rhs.isIdealRefraction;
            albedoMap = rhs.albedoMap;
            normalMap = rhs.normalMap;
            roughnessMap = rhs.roughnessMap;

            if (type == MaterialType::CarPaint) {
                carpaint = rhs.carpaint;
            }
            else if (type == MaterialType::Retroreflective) {
                retrorelective = rhs.retrorelective;
            }
            else {
                standard = rhs.standard;
            }

            return *this;
        }
    };

    class IMaterialParamEditor {
    protected:
        IMaterialParamEditor() {}
        virtual ~IMaterialParamEditor() {}

    public:
        virtual bool edit(std::string_view name, float& param, float _min = 0.0F, float _max = 1.0F) = 0;
        virtual bool edit(std::string_view name, vec3& param) = 0;
        virtual bool edit(std::string_view name, vec4& param) = 0;

        void editTex(std::string_view name, int32_t texid)
        {
            if (texid >= 0) {
                /*auto tex = aten::texture::GtTexture(texid);
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
        real pdf{ real(0) };

        AT_DEVICE_API MaterialSampling() {}
        AT_DEVICE_API MaterialSampling(const aten::vec3& d, const aten::vec3& b, real p)
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
            real ior = 1)
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

        static std::shared_ptr<material> CreateMaterialWithDefaultValue(aten::MaterialType type);

        static std::shared_ptr<material> CreateMaterialWithMaterialParameter(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        material() = default;
        virtual ~material() = default;

        bool isEmissive() const
        {
            return m_param.attrib.isEmissive;
        }

        bool isSingular() const
        {
            return m_param.attrib.isSingular;
        }

        bool isTranslucent() const
        {
            return m_param.attrib.isTranslucent;
        }

        bool isSingularOrTranslucent() const
        {
            return isSingular() || isTranslucent();
        }

        // TODO
        virtual bool isGlossy() const
        {
            bool isGlossy = m_param.attrib.isGlossy;

            if (isGlossy) {
                isGlossy = (m_param.standard.roughness == real(1) ? false : true);
                if (!isGlossy) {
                    isGlossy = (m_param.standard.shininess == 0 ? false : true);
                }
            }

            return isGlossy;
        }

        const aten::vec4& color() const
        {
            return m_param.baseColor;
        }

        int32_t id() const
        {
            return m_param.id;
        }

        void setTextures(
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        real ior() const
        {
            return m_param.standard.ior;
        }

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
            real u, real v);

        static AT_DEVICE_API real getTranslucentAlpha(
            const aten::MaterialParameter& param,
            real u, real v);

        static AT_DEVICE_API real computeFresnel(
            real ni, real nt,
            const aten::vec3& wi,
            const aten::vec3& normal)
        {
            const auto cosi = dot(normal, wi);

            const auto nnt = ni / nt;
            const auto sini2 = real(1.0) - cosi * cosi;
            const auto sint2 = nnt * nnt * sini2;
            const auto cost = aten::sqrt(aten::cmpMax(real(0.0), real(1.0) - sint2));

            const auto rp = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
            const auto rs = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);

            const auto Rsp = (rp * rp + rs * rs) * real(0.5);
            return Rsp;
        }

        /**
         * @brief Compute schlick's fresnel.
         * @param[in] ior Refraction index.
         * @param[in] wo Output vector.
         * @param[in] n Normal vector on surface.
         * @return fresnel.
         */
        static inline AT_DEVICE_API float ComputeSchlickFresnel(
            const float ior,
            const aten::vec3& wo,
            const aten::vec3& n)
        {
            // NOTE:
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE:
            // F_schlick(v,h) ≒ f0 + (1 - f0)(1 - cos_theta)^5
            // f0 = pow((1 - ior) / (1 + ior), 2)
            // ior = nt / ni
            // f0 = pow((1 - ior) / (1 + ior), 2)
            //    = pow((1 - nt/ni) / (1 + nt/ni), 2)
            //    = pow((ni - nt) / (ni + nt), 2)
            auto f0 = (1.0f - ior) / (1.0f + ior);
            f0 = f0 * f0;

            const auto LN = aten::abs(dot(wo, n));

            const auto fresnel = f0 + (1 - f0) * aten::pow((1 - LN), 5);
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

        static AT_DEVICE_API aten::vec4 sampleAlbedoMap(
            const aten::MaterialParameter* mtrl,
            real u, real v,
            uint32_t lod = 0);

        static AT_DEVICE_API void sampleMaterial(
            AT_NAME::MaterialSampling* result,
            const aten::MaterialParameter* mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real pre_sampled_r,
#ifdef __CUDACC__
            float u, float v);
#else
            float u, float v,
            bool is_light_path = false);
#endif

        static AT_DEVICE_API real samplePDF(
            const aten::MaterialParameter* dst_mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v);

        static AT_DEVICE_API aten::vec3 sampleDirection(
            const aten::MaterialParameter* dst_mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            real u, real v,
            aten::sampler* sampler,
            real pre_sampled_r);

        static AT_DEVICE_API aten::vec3 sampleBSDF(
            const aten::MaterialParameter* dst_mtrl,
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v,
            real pre_sampled_r);

        static AT_DEVICE_API real applyNormal(
            const aten::MaterialParameter* mtrl,
            const int32_t normalMapIdx,
            const aten::vec3& orgNml,
            aten::vec3& newNml,
            real u, real v,
            const aten::vec3& wi,
            aten::sampler* sampler);

    protected:
        aten::MaterialParameter m_param;

        // For debug.
        std::string m_name;
    };
}
