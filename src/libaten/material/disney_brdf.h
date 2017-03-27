#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten
{
	class DisneyBRDF : public material {
	public:
		struct Parameter {
			vec3 baseColor{ vec3(0.82, 0.67, 0.16) };
			real metallic{ 0 };
			real subsurface{ 0 };
			real specular{ 0.5 };
			real roughness{ 0.5 };
			real specularTint{ 0 };
			real anisotropic{ 0 };
			real sheen{ 0 };
			real sheenTint{ 0.5 };
			real clearcoat{ 0 };
			real clearcoatGloss{ 1 };
		};
	public:
		DisneyBRDF() {}
		DisneyBRDF(
			vec3 baseColor,
			real subsurface,
			real metallic,
			real specular,
			real specularTint,
			real roughness,
			real anisotropic,
			real sheen,
			real sheenTint,
			real clearcoat,
			real clearcoatGloss,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(baseColor, 1, albedoMap, normalMap)
		{
			m_baseColor = baseColor;
			m_subsurface = aten::clamp<real>(subsurface, 0, 1);
			m_metallic = aten::clamp<real>(metallic, 0, 1);
			m_specular = aten::clamp<real>(specular, 0, 1);
			m_specularTint = aten::clamp<real>(specularTint, 0, 1);
			m_roughness = aten::clamp<real>(roughness, 0, 1);
			m_anisotropic = aten::clamp<real>(anisotropic, 0, 1);
			m_sheen = aten::clamp<real>(sheen, 0, 1);
			m_sheenTint = aten::clamp<real>(sheenTint, 0, 1);
			m_clearcoat = aten::clamp<real>(clearcoat, 0, 1);
			m_clearcoatGloss = aten::clamp<real>(clearcoatGloss, 0, 1);

			m_roughnessMap = roughnessMap;
		}

		DisneyBRDF(
			const Parameter& param,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(param.baseColor, 1, albedoMap, normalMap)
		{
			m_baseColor = param.baseColor;
			m_subsurface = aten::clamp<real>(param.subsurface, 0, 1);
			m_metallic = aten::clamp<real>(param.metallic, 0, 1);
			m_specular = aten::clamp<real>(param.specular, 0, 1);
			m_specularTint = aten::clamp<real>(param.specularTint, 0, 1);
			m_roughness = aten::clamp<real>(param.roughness, 0, 1);
			m_anisotropic = aten::clamp<real>(param.anisotropic, 0, 1);
			m_sheen = aten::clamp<real>(param.sheen, 0, 1);
			m_sheenTint = aten::clamp<real>(param.sheenTint, 0, 1);
			m_clearcoat = aten::clamp<real>(param.clearcoat, 0, 1);
			m_clearcoatGloss = aten::clamp<real>(param.clearcoatGloss, 0, 1);

			m_roughnessMap = roughnessMap;
		}

		DisneyBRDF(Values& val)
			: material(val)
		{
			// TODO
			// Clamp parameters.
			m_subsurface = val.get("subsurface", m_subsurface);
			m_metallic = val.get("metallic", m_metallic);
			m_specular = val.get("specular", m_specular);
			m_specularTint = val.get("specularTint", m_specularTint);
			m_roughness = val.get("roughness", m_roughness);
			m_anisotropic = val.get("anisotropic", m_anisotropic);
			m_sheen = val.get("sheen", m_sheen);
			m_sheenTint = val.get("sheenTint", m_sheenTint);
			m_clearcoat = val.get("clearcoat", m_clearcoat);
			m_clearcoatGloss = val.get("clearcoatGloss", m_clearcoatGloss);
			m_roughnessMap = val.get("roughnessmap", m_roughnessMap);
		}

		virtual ~DisneyBRDF() {}

	public:
		virtual bool isGlossy() const override final
		{
			return (m_roughness == 1 ? false : true);
		}

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal,
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual sampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const override final;

	private:
		real pdf(
			const vec3& V,
			const vec3& N,
			const vec3& L,
			const vec3& X,
			const vec3& Y,
			real u, real v) const;

		vec3 sampleDirection(
			const vec3& V,
			const vec3& N,
			const vec3& X,
			const vec3& Y,
			real u, real v,
			sampler* sampler) const;

		vec3 bsdf(
			real& fresnel,
			const vec3& V,
			const vec3& N,
			const vec3& L,
			const vec3& X,
			const vec3& Y,
			real u, real v) const;

	private:
		vec3 m_baseColor;		// サーフェイスカラー，通常テクスチャマップによって供給される.
		real m_subsurface;		// 表面下の近似を用いてディフューズ形状を制御する.
		real m_metallic;		// 金属度(0 = 誘電体, 1 = 金属)。これは2つの異なるモデルの線形ブレンドです。金属モデルはディフューズコンポーネントを持たず，また色合い付けされた入射スペキュラーを持ち，基本色に等しくなります.
		real m_specular;		// 入射鏡面反射量。これは明示的な屈折率の代わりにあります.
		real m_specularTint;	// 入射スペキュラーを基本色に向かう色合いをアーティスティックな制御するための譲歩。グレージングスペキュラーはアクロマティックのままです.
		real m_roughness;		// 表面の粗さで，ディフューズとスペキュラーレスポンスの両方を制御します.
		real m_anisotropic;		// 異方性の度合い。これはスペキュラーハイライトのアスペクト比を制御します(0 = 等方性, 1 = 最大異方性).
		real m_sheen;			// 追加的なグレージングコンポーネント，主に布に対して意図している.
		real m_sheenTint;		// 基本色に向かう光沢色合いの量.
		real m_clearcoat;		// 第二の特別な目的のスペキュラーローブ.
		real m_clearcoatGloss;	// クリアコートの光沢度を制御する(0 = “サテン”風, 1 = “グロス”風).

		texture* m_roughnessMap{ nullptr };
	};
}
