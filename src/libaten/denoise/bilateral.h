#pragma once

#include "visualizer/visualizer.h"

namespace aten {
	class BilateralFilter: public visualizer::PreProc{
	public:
		BilateralFilter() {}
		BilateralFilter(real sigmaS, real sigmaR)
		{
			setParam(sigmaS, sigmaR);
		}

		virtual ~BilateralFilter() {}

	public:
		virtual void operator()(
			const vec3* src,
			uint32_t width, uint32_t height,
			vec3* dst) override final;

		void setParam(real sigmaS, real sigmaR)
		{
			m_sigmaS = sigmaS;
			m_sigmaR = sigmaR;
		}

	private:
		real m_sigmaS{ 0.2 };
		real m_sigmaR{ 0.2 };
	};
}
