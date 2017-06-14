#pragma once

#include "renderer/background.h"
#include "texture/texture.h"

namespace AT_NAME
{
	class envmap : public aten::background {
	public:
		envmap() {}
		virtual ~envmap() {}

	public:
		void init(
			aten::texture* envmap,
			real mult = real(1))
		{
			m_envmap = envmap;
			m_mult = mult;
		}

		virtual aten::vec3 sample(const aten::ray& inRay) const override final;

		virtual aten::vec3 sample(real u, real v) const override final;

		const aten::texture* getTexture() const
		{
			return m_envmap;
		}

		static AT_DEVICE_API aten::vec3 envmap::convertUVToDirection(real u, real v)
		{
			// u = phi / 2PI
			// => phi = 2PI * u;
			auto phi = 2 * AT_MATH_PI * u;

			// v = 1 - theta / PI
			// => theta = (1 - v) * PI;
			auto theta = (1 - v) * AT_MATH_PI;

			aten::vec3 dir;

			dir.y = aten::cos(theta);

			auto xz = aten::sqrt(1 - dir.y * dir.y);

			dir.x = xz * aten::sin(phi);
			dir.z = xz * aten::cos(phi);

			// ”O‚Ì‚½‚ß...
			dir = normalize(dir);

			return std::move(dir);
		}

		static AT_DEVICE_API aten::vec3 envmap::convertDirectionToUV(const aten::vec3& dir)
		{
			auto temp = aten::atan2(dir.x, dir.z);
			auto r = length(dir);

			// Account for discontinuity
			auto phi = (real)((temp >= 0) ? temp : (temp + 2 * AT_MATH_PI));
			auto theta = aten::acos(dir.y / r);

			// Map to [0,1]x[0,1] range and reverse Y axis
			real u = phi / (2 * AT_MATH_PI);
			real v = 1 - theta / AT_MATH_PI;

			aten::vec3 uv = aten::vec3(u, v, 0);

			return std::move(uv);
		}
		
	private:
		aten::texture* m_envmap;
		real m_mult{ real(1) };
	};
}
