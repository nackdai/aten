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
		float baseColor[3];
		float ior{ 1.0f };
		float roughness{ 0.0f };
		float shininess{ 0.0f };
#if 0
		// TODO
		float subsurface{ 0.0f };
		float metallic{ 0.0f };
		float specular{ 0.0f };
		float specularTint{ 0.0f };
		float anisotropic{ 0.0f };
		float sheen{ 0.0f };
		float sheenTint{ 0.0f };
		float clearcoat{ 0.0f };
		float clearcoatGloss{ 0.0f };
#endif
		int albedoMap{ -1 };
		int normalMap{ -1 };
		int roughnessMap{ -1 };
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
			: m_albedo(clr), m_ior(ior), m_albedoMap(albedoMap), m_normalMap(normalMap)
		{}
		material(Values& val)
		{
			m_albedo = val.get("color", m_albedo);
			m_ior = val.get("ior", m_ior);
			m_albedoMap = (texture*)val.get("albedomap", (void*)m_albedoMap);
			m_normalMap = (texture*)val.get("normalmap", (void*)m_normalMap);
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
			return m_albedo;
		}

		uint32_t id() const
		{
			return m_id;
		}

		virtual vec3 sampleAlbedoMap(real u, real v) const
		{
			vec3 albedo(1, 1, 1);
			if (m_albedoMap) {
				albedo = m_albedoMap->at(u, v);
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

		virtual void serialize(MaterialParam& param) const = 0;

		real ior() const
		{
			return m_ior;
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

	private:
		uint32_t m_id{ 0 };

		vec3 m_albedo;

		real m_ior{ 1 };

		texture* m_albedoMap{ nullptr };
		texture* m_normalMap{ nullptr };
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
