#pragma once

#include "kernel/renderer.h"

namespace idaten
{
	template <class BASE>
	class RendererMultiGPU : public BASE {
	protected:
		RendererMultiGPU() {}
		virtual ~RendererMultiGPU() {}

	public:
		virtual void postRender(int width, int height) = 0;
	};
}
