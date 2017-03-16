#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"

namespace aten
{
	struct hitrecord;

	class material {
		friend class LayeredBSDF;

	public:
		material() {}
		virtual ~material() {}

	protected:
		material(
			const vec3& clr, 
			real ior = 1,
			texture* albedoMap = nullptr, 
			texture* normalMap = nullptr) 
			: m_albedo(clr), m_ior(ior), m_albedoMap(albedoMap), m_normalMap(normalMap)
		{}

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

		const vec3& color() const
		{
			return m_albedo;
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
			real u, real v,
			sampler* sampler) const = 0;

		virtual vec3 sampleDirection(
			const vec3& in,
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

			sampling() {}
			sampling(const vec3& d, const vec3& b, real p)
				: dir(d), bsdf(b), pdf(p)
			{}
		};

		virtual sampling sample(
			const vec3& in,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const = 0;

	protected:
		real ior() const
		{
			return m_ior;
		}

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

	private:
		vec3 m_albedo;

		real m_ior{ 0 };

		texture* m_albedoMap{ nullptr };
		texture* m_normalMap{ nullptr };
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
