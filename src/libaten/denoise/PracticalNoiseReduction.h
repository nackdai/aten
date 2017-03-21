#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

namespace aten {
	class PracticalNoiseReduction : public visualizer::PreProc{
	public:
		PracticalNoiseReduction() {}

		virtual ~PracticalNoiseReduction() {}

	public:
		virtual void operator()(
			const vec4* src,
			uint32_t width, uint32_t height,
			vec4* dst) override final;

		void setBuffers(
			vec4* direct,
			vec4* indirect,
			vec4* varIndirect,
			vec4* nml_depth)
		{
			m_direct = direct;
			m_indirect = indirect;
			m_variance = varIndirect;
			m_nml_depth = nml_depth;
		}

	private:
		vec4* m_direct{ nullptr };

		vec4* m_indirect{ nullptr };
		vec4* m_variance{ nullptr };

		vec4* m_nml_depth{ nullptr };
	};
}
