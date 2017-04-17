#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"
#include "misc/value.h"
#include "math/ray.h"

namespace aten
{
	struct hitrecord;

	struct MaterialParam {
		// サーフェイスカラー，通常テクスチャマップによって供給される.
		union {
			vec3 baseColor;
			float baseColorArray[3];
		};

		float ior{ 1.0f };
		float roughness{ 0.0f };			// 表面の粗さで，ディフューズとスペキュラーレスポンスの両方を制御します.
		float shininess{ 0.0f };

		float subsurface{ 0.0f };			// 表面下の近似を用いてディフューズ形状を制御する.
		float metallic{ 0.0f };				// 金属度(0 = 誘電体, 1 = 金属)。これは2つの異なるモデルの線形ブレンドです。金属モデルはディフューズコンポーネントを持たず，また色合い付けされた入射スペキュラーを持ち，基本色に等しくなります.
		float specular{ 0.0f };				// 入射鏡面反射量。これは明示的な屈折率の代わりにあります.
		float specularTint{ 0.0f };			// 入射スペキュラーを基本色に向かう色合いをアーティスティックな制御するための譲歩。グレージングスペキュラーはアクロマティックのままです.
		float anisotropic{ 0.0f };			// 異方性の度合い。これはスペキュラーハイライトのアスペクト比を制御します(0 = 等方性, 1 = 最大異方性).
		float sheen{ 0.0f };				// 追加的なグレージングコンポーネント，主に布に対して意図している.
		float sheenTint{ 0.0f };			// 基本色に向かう光沢色合いの量.
		float clearcoat{ 0.0f };			// 第二の特別な目的のスペキュラーローブ.
		float clearcoatGloss{ 0.0f };		// クリアコートの光沢度を制御する(0 = “サテン”風, 1 = “グロス”風).

		union {								
			int albedoMapIdx;
			void* albedoMap{ nullptr };
		};

		union {
			int normalMapIdx;
			void* normalMap{ nullptr };
		};

		union {
			int roughnessMapIdx;
			void* roughnessMap{ nullptr };
		};

		MaterialParam() {}
	};

	class material {
		friend class LayeredBSDF;

	protected:
		material();
		virtual ~material() {}

	protected:
		material(
			const vec3& clr, 
			real ior = 1,
			texture* albedoMap = nullptr, 
			texture* normalMap = nullptr) 
		{
			m_param.baseColor = clr;
			m_param.ior = ior;
			m_param.albedoMap = albedoMap;
			m_param.normalMap = normalMap;
		}
		material(Values& val)
		{
			m_param.baseColor = val.get("color", m_param.baseColor);
			m_param.ior = val.get("ior", m_param.ior);
			m_param.albedoMap = (texture*)val.get("albedomap", (void*)m_param.albedoMap);
			m_param.normalMap = (texture*)val.get("normalmap", (void*)m_param.normalMap);
		}

	public:
		virtual bool isEmissive() const
		{
			return false;
		}

		virtual bool isSingular() const
		{
			return false;
		}

		virtual bool isTranslucent() const
		{
			return false;
		}

		virtual bool isGlossy() const
		{
			return false;
		}

		virtual bool isNPR() const
		{
			return false;
		}

		const vec3& color() const
		{
			return m_param.baseColor;
		}

		uint32_t id() const
		{
			return m_id;
		}

		virtual vec3 sampleAlbedoMap(real u, real v) const
		{
			vec3 albedo(1, 1, 1);
			if (m_param.albedoMap) {
				albedo = ((texture*)m_param.albedoMap)->at(u, v);
			}
			return std::move(albedo);
		}

		virtual void applyNormalMap(
			const vec3& orgNml,
			vec3& newNml,
			real u, real v) const;

		virtual real computeFresnel(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real outsideIor = 1) const;

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const = 0;

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal, 
			real u, real v,
			sampler* sampler) const = 0;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const = 0;

		struct sampling {
			vec3 dir;
			vec3 bsdf;
			real pdf{ real(0) };
			real fresnel{ real(1) };

			real subpdf{ real(1) };

			sampling() {}
			sampling(const vec3& d, const vec3& b, real p)
				: dir(d), bsdf(b), pdf(p)
			{}
		};

		virtual sampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false) const = 0;

		real ior() const
		{
			return m_param.ior;
		}

	protected:
		static vec3 sampleTexture(texture* tex, real u, real v, real defaultValue)
		{
			auto ret = sampleTexture(tex, u, v, vec3(defaultValue));
			return std::move(ret);
		}

		static vec3 sampleTexture(texture* tex, real u, real v, const vec3& defaultValue)
		{
			vec3 ret = defaultValue;
			if (tex) {
				ret = tex->at(u, v);
			}
			return std::move(ret);
		}

		static void serialize(const material* mtrl, MaterialParam& param);

	protected:
		uint32_t m_id{ 0 };

		MaterialParam m_param;
	};

	class Light;

	class NPRMaterial : public material {
	protected:
		NPRMaterial() {}
		NPRMaterial(const vec3& e, Light* light);

		NPRMaterial(Values& val)
			: material(val)
		{}

		virtual ~NPRMaterial() {}

	public:
		virtual bool isNPR() const override final
		{
			return true;
		}

		virtual real computeFresnel(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real outsideIor = 1) const override final
		{
			return real(1);
		}

		void setTargetLight(Light* light);

		const Light* getTargetLight() const;

		virtual vec3 bsdf(
			real cosShadow,
			real u, real v) const = 0;

	private:
		Light* m_targetLight{ nullptr };
	};


	real schlick(
		const vec3& in,
		const vec3& normal,
		real ni, real nt);

	real computFresnel(
		const vec3& in,
		const vec3& normal,
		real ni, real nt);
}
