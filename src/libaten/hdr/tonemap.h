#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/shader.h"

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
			color* dst);
	};

	class TonemapRender : public SimpleRender {
	public:
		TonemapRender() {}
		virtual ~TonemapRender() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;
	};
}