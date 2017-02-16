#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten
{
	class lambert : public material {
	public:
		lambert() {}
		lambert(const vec3& c, texture* tex = nullptr)
			: m_color(c)
		{
			settexture(tex);
		}

		virtual ~lambert() {}

		virtual vec3 color() const override final
		{
			return m_color;
		}

		virtual real pdf(const vec3& normal, const vec3& dir) const override final;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal,
			sampler* sampler) const override final;

		virtual vec3 brdf(
			const vec3& normal, const vec3& dir,
			real u, real v) const override final;

		virtual sampling sample(
			const vec3& in,
			const vec3& normal,
			sampler* sampler,
			real u, real v) const override final;

	private:
		vec3 m_color{ vec3(1, 1, 1) };
	};
}
