#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
	class Tonemap {
	public:
		static std::tuple<real, real> computeAvgAndMaxLum(
			int width, int height,
			const vec3* src);

		static void doTonemap(
			int width, int height,
			const vec3* src,
			TColor<uint8_t>* dst);
	};

	class TonemapRender : public Blitter {
	public:
		TonemapRender() {}
		virtual ~TonemapRender() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;
	};
}