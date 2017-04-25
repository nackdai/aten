#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"
#include "misc/value.h"
#include "math/ray.h"

namespace AT_NAME {
	class Light;
}

namespace aten
{
	struct hitrecord;

	struct MaterialAttribute {
		struct {
			const uint32_t isEmissive : 1;
			const uint32_t isSingular : 1;
			const uint32_t isTranslucent : 1;
			const uint32_t isGlossy : 1;
			const uint32_t isNPR : 1;
		};

		MaterialAttribute(
			bool _isEmissive = false,
			bool _isSingular = false,
			bool _isTranslucent = false,
			bool _isGlossy = false,
			bool _isNPR = false)
			: isEmissive(_isEmissive), isSingular(_isSingular), isTranslucent(_isTranslucent),
			isGlossy(_isGlossy), isNPR(_isNPR)
		{}
		MaterialAttribute(const MaterialAttribute& type)
			: MaterialAttribute(type.isEmissive, type.isSingular, type.isTranslucent, type.isGlossy, type.isNPR)
		{}
	};

	//													          Em     Sp      Tr    Gl    NPR
	#define MaterialAttributeMicrofacet	aten::MaterialAttribute(false, false, false, true,  false)
	#define MaterialAttributeLambert	aten::MaterialAttribute(false, false, false, false, false)
	#define MaterialAttributeEmissive	aten::MaterialAttribute(true,  false, false, false, false)
	#define MaterialAttributeSpecular	aten::MaterialAttribute(false, true,  false, true,  false)
	#define MaterialAttributeRefraction	aten::MaterialAttribute(false, true,  true,  true,  false)
	#define MaterialAttributeNPR		aten::MaterialAttribute(false, false, false, false, true)

	enum MaterialType {
		Emissive,
		Lambert,
		OrneNayar,
		Specular,
		Refraction,
		Blinn,
		GGX,
		Beckman,
		Disney,
		Toon,
		Layer,

		MaterialTypeMax,
	};

	struct MaterialParameter {
		MaterialType type;

		aten::vec3 baseColor;					// サーフェイスカラー，通常テクスチャマップによって供給される.

		real ior{ 1.0f };
		real roughness{ 0.0f };			// 表面の粗さで，ディフューズとスペキュラーレスポンスの両方を制御します.
		real shininess{ 1.0f };

		real subsurface{ 0.0f };		// 表面下の近似を用いてディフューズ形状を制御する.
		real metallic{ 0.0f };			// 金属度(0 = 誘電体, 1 = 金属)。これは2つの異なるモデルの線形ブレンドです。金属モデルはディフューズコンポーネントを持たず，また色合い付けされた入射スペキュラーを持ち，基本色に等しくなります.
		real specular{ 0.0f };			// 入射鏡面反射量。これは明示的な屈折率の代わりにあります.
		real specularTint{ 0.0f };		// 入射スペキュラーを基本色に向かう色合いをアーティスティックな制御するための譲歩。グレージングスペキュラーはアクロマティックのままです.
		real anisotropic{ 0.0f };		// 異方性の度合い。これはスペキュラーハイライトのアスペクト比を制御します(0 = 等方性, 1 = 最大異方性).
		real sheen{ 0.0f };				// 追加的なグレージングコンポーネント，主に布に対して意図している.
		real sheenTint{ 0.0f };			// 基本色に向かう光沢色合いの量.
		real clearcoat{ 0.0f };			// 第二の特別な目的のスペキュラーローブ.
		real clearcoatGloss{ 0.0f };	// クリアコートの光沢度を制御する(0 = “サテン”風, 1 = “グロス”風).

		UnionIdxPtr albedoMap;
		UnionIdxPtr normalMap;
		UnionIdxPtr roughnessMap;

		bool isIdealRefraction{ false };

		const MaterialAttribute attrib;

		MaterialParameter(MaterialType _type, const MaterialAttribute& _attrib)
			: type(_type), attrib(_attrib)
		{}
	};
}

namespace AT_NAME
{
	struct MaterialSampling {
		aten::vec3 dir;
		aten::vec3 bsdf;
		real pdf{ real(0) };
		real fresnel{ real(1) };

		real subpdf{ real(1) };

		AT_DEVICE_API MaterialSampling() {}
		AT_DEVICE_API MaterialSampling(const aten::vec3& d, const aten::vec3& b, real p)
			: dir(d), bsdf(b), pdf(p)
		{}
	};

	class material {
		friend class LayeredBSDF;

		static std::vector<material*> g_materials;

	protected:
		material(aten::MaterialType type, const aten::MaterialAttribute& attrib);
		virtual ~material();

		material(
			aten::MaterialType type,
			const aten::MaterialAttribute& attrib,
			const aten::vec3& clr,
			real ior = 1,
			aten::texture* albedoMap = nullptr,
			aten::texture* normalMap = nullptr);

		material(aten::MaterialType type, const aten::MaterialAttribute& attrib, aten::Values& val);

	public:
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

		// TODO
		virtual bool isGlossy() const
		{
			bool isGlossy = m_param.attrib.isGlossy;

			if (isGlossy) {
				isGlossy = (m_param.roughness == real(1) ? false : true);
				if (!isGlossy) {
					isGlossy = (m_param.shininess == 0 ? false : true);
				}
			}

			return isGlossy;
		}

		bool isNPR() const
		{
			return m_param.attrib.isNPR;
		}

		const aten::vec3& color() const
		{
			return m_param.baseColor;
		}

		uint32_t id() const
		{
			return m_id;
		}

		virtual aten::vec3 sampleAlbedoMap(real u, real v) const
		{
			return std::move(sampleTexture((const aten::texture*)m_param.albedoMap.ptr, u, v, real(1)));
		}

		virtual void applyNormalMap(
			const aten::vec3& orgNml,
			aten::vec3& newNml,
			real u, real v) const;

		virtual real computeFresnel(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor = 1) const;

		virtual real pdf(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const = 0;

		virtual aten::vec3 sampleDirection(
			const aten::ray& ray,
			const aten::vec3& normal,
			real u, real v,
			aten::sampler* sampler) const = 0;

		virtual aten::vec3 bsdf(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real u, real v) const = 0;

		virtual MaterialSampling sample(
			const aten::ray& ray,
			const aten::vec3& normal,
			const aten::hitrecord& hitrec,
			aten::sampler* sampler,
			real u, real v,
			bool isLightPath = false) const = 0;

		real ior() const
		{
			return m_param.ior;
		}

		const aten::MaterialParameter& param() const
		{
			return m_param;
		}

		static AT_DEVICE_API aten::vec3 sampleTexture(const aten::texture* tex, real u, real v, real defaultValue)
		{
			auto ret = sampleTexture(tex, u, v, aten::vec3(defaultValue));
			return std::move(ret);
		}

		static AT_DEVICE_API aten::vec3 sampleTexture(const aten::texture* tex, real u, real v, const aten::vec3& defaultValue)
		{
			aten::vec3 ret = defaultValue;

			// TODO
#ifndef __AT_CUDA__
			if (tex) {
				ret = tex->at(u, v);
			}
#endif

			return std::move(ret);
		}

		static uint32_t getMaterialNum();
		static const material* getMaterial(uint32_t idx);
		static int findMaterialIdx(material* mtrl);
		static const std::vector<material*>& getMaterials();

	protected:
		uint32_t m_id{ 0 };

		aten::MaterialParameter m_param;
	};

	class NPRMaterial : public material {
	protected:
		NPRMaterial(
			aten::MaterialType type,
			const aten::vec3& e, AT_NAME::Light* light);

		NPRMaterial(aten::MaterialType type, aten::Values& val)
			: material(type, MaterialAttributeNPR, val)
		{}

		virtual ~NPRMaterial() {}

	public:
		virtual real computeFresnel(
			const aten::vec3& normal,
			const aten::vec3& wi,
			const aten::vec3& wo,
			real outsideIor = 1) const override final
		{
			return real(1);
		}

		void setTargetLight(AT_NAME::Light* light);

		const AT_NAME::Light* getTargetLight() const;

		virtual aten::vec3 bsdf(
			real cosShadow,
			real u, real v) const = 0;

	private:
		AT_NAME::Light* m_targetLight{ nullptr };
	};


	real schlick(
		const aten::vec3& in,
		const aten::vec3& normal,
		real ni, real nt);

	real computFresnel(
		const aten::vec3& in,
		const aten::vec3& normal,
		real ni, real nt);
}
