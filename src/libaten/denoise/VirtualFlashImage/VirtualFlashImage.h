#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

namespace aten {
	class VirtualFlashImage : public visualizer::PreProc {
	public:
		VirtualFlashImage() {}
		virtual ~VirtualFlashImage() {}

	public:
		virtual void operator()(
			const vec4* src,
			uint32_t width, uint32_t height,
			vec4* dst) override final;

		void setParam(
			uint32_t numSamples,
			vec4* flash,
			vec4* varFlash)
		{
			m_numSamples = numSamples;
			m_flash = flash;
			m_varFlash = varFlash;
		}

	private:
		uint32_t m_numSamples;

		vec4* m_flash;
		vec4* m_varFlash;	// variance of flash image.
	};
}
