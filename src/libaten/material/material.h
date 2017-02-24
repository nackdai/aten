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

		virtual vec3 color() const = 0;

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
