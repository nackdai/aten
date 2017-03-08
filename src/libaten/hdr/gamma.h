#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
	class GammaCorrection : public Blitter {
	public:
		GammaCorrection() {}
		GammaCorrection(float gamma)
		{
			m_gamma = std::max(1.0f, gamma);
		}
		virtual ~GammaCorrection() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;

	private:
		float m_gamma{ 2.2f };
	};
}