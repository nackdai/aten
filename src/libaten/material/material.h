#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"

namespace aten
{
	struct hitrecord;

	class material {
	public:
		material() {}
		virtual ~material() {}

	protected:
		material(
			const vec3& clr, 
			texture* albedoMap = nullptr, 
			texture* normalMap = nullptr) 
			: m_albedo(clr), m_albedoMap(albedoMap), m_normalMap(normalMap)
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

		vec3 sampleAlbedoMap(real u, real v) const
		{
			vec3 albedo(1, 1, 1);
			if (m_albedoMap) {
				albedo = m_albedoMap->at(u, v);
			}
			return std::move(albedo);
		}
		void applyNormalMap(
			const vec3& orgNml,
			vec3& newNml,
			real u, real v) const;

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo) const = 0;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal, 
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

	private:
		vec3 m_albedo;

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
