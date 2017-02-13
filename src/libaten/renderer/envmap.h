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
			real mult = CONST_REAL(1.0))
		{
			m_envmap = envmap;
			m_mult = mult;
		}

		virtual vec3 sample(const ray& inRay) const override final;

		virtual vec3 sample(real u, real v) const override final;
		
	private:
		texture* m_envmap;
		real m_mult{ CONST_REAL(1.0) };
	};
}
