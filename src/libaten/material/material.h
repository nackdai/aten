#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "texture/texture.h"

namespace aten
{
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

		virtual real pdf(const vec3& normal, const vec3& dir) const = 0;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal, 
			sampler* sampler) const = 0;

		virtual vec3 brdf(
			const vec3& normal, 
			const vec3& dir,
			real u, real v) const = 0;

		struct sampling {
			vec3 dir;
			vec3 brdf;
			real pdf{ real(0) };
			bool into{ false };

			sampling() {}
			sampling(const vec3& d, const vec3& b, real p)
				: dir(d), brdf(b), pdf(p)
			{}
		};

		virtual sampling sample(
			const vec3& in,
			const vec3& normal,
			sampler* sampler,
			real u, real v) const = 0;

		void settexture(texture* tex)
		{
			m_tex = tex;
		}

		const texture* tex() const
		{
			return m_tex;
		}

	private:
		texture* m_tex{ nullptr };
	};
}
