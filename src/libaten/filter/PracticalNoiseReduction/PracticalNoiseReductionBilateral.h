#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

namespace aten {
	class PracticalNoiseReductionBilateralFilter {
	public:
		PracticalNoiseReductionBilateralFilter() {}
		PracticalNoiseReductionBilateralFilter(real sigmaS, real sigmaR, real sigmaD)
		{
			setParam(sigmaS, sigmaR, sigmaD);
		}

		virtual ~PracticalNoiseReductionBilateralFilter() {}

	public:
		void operator()(
			const vec4* src,
			const vec4* nml_depth,
			uint32_t width, uint32_t height,
			vec4* dst,
			vec4* variance);

		void setParam(real sigmaS, real sigmaR, real sigmaD)
		{
			m_sigmaS = sigmaS;
			m_sigmaR = sigmaR;
			m_sigmaD = sigmaD;
		}

	private:
		real m_sigmaS{ 0.2 };
		real m_sigmaR{ 0.2 };
		real m_sigmaD{ 0.02 };
	};
}
