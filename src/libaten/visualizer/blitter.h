#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/visualizer.h"

namespace aten {
	class Blitter : public visualizer::PostProc {
	public:
		Blitter() {}
		virtual ~Blitter() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override;

		virtual PixelFormat inFormat() const override
		{
			return PixelFormat::rgba32f;
		}
		virtual PixelFormat outFormat() const override
		{
			return PixelFormat::rgba32f;
		}
	};

}