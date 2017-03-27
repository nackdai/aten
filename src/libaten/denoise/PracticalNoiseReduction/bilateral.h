#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

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
			const vec4* src,
			uint32_t width, uint32_t height,
			vec4* dst) override final;

		void setParam(real sigmaS, real sigmaR)
		{
			m_sigmaS = sigmaS;
			m_sigmaR = sigmaR;
		}

		void setVarianceBuffer(vec4* v)
		{
			m_variance = v;
		}

	private:
		real m_sigmaS{ 0.2 };
		real m_sigmaR{ 0.2 };

		vec4* m_variance{ nullptr };
	};

	class BilateralFilterShader : public Blitter {
	public:
		BilateralFilterShader() {}
		BilateralFilterShader(real sigmaS, real sigmaR)
		{
			setParam(sigmaS, sigmaR);
		}

		virtual ~BilateralFilterShader() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;

		void setParam(real sigmaS, real sigmaR)
		{
			m_sigmaS = sigmaS;
			m_sigmaR = sigmaR;
		}

	private:
		real m_sigmaS{ 0.2 };
		real m_sigmaR{ 0.2 };

		static const uint32_t buffersize = 10;
		float distW[buffersize + 1][buffersize + 1];

		int m_radius{ 0 };
	};
}
