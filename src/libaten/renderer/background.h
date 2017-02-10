#pragma once

#include "types.h"
#include "math/vec3.h"
#include "renderer/ray.h"

namespace aten
{
	class background {
	protected:
		background() {}
		virtual ~background() {}

	public:
		virtual vec3 sample(const ray& inRay) const = 0;
	};

	class StaticColorBG : public background {
	public:
		StaticColorBG(const vec3 c)
			: m_color(c)
		{}
		virtual ~StaticColorBG() {}

	public:
		virtual vec3 sample(const ray& inRay) const override final
		{
			return std::move(m_color);
		}

	private:
		vec3 m_color;
	};
}
