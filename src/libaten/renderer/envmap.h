#pragma once

#include "renderer/background.h"
#include "texture/texture.h"

namespace aten
{
	class envmap : public background {
	public:
		envmap() {}
		virtual ~envmap() {}

	public:
		void init(
			texture* envmap,
			real mult = real(1))
		{
			m_envmap = envmap;
			m_mult = mult;
		}

		virtual vec3 sample(const ray& inRay) const override final;

		virtual vec3 sample(real u, real v) const override final;

		const texture* getTexture() const
		{
			return m_envmap;
		}

		static vec3 convertUVToDirection(real u, real v);
		static vec3 convertDirectionToUV(const vec3& dir);
		
	private:
		texture* m_envmap;
		real m_mult{ real(1) };
	};
}
