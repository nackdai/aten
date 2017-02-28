#pragma once

#include "visualizer/visualizer.h"

namespace aten {
	class NonLocalMeanFilter : public visualizer::PreProc {
	public:
		NonLocalMeanFilter() {}
		NonLocalMeanFilter(real param_h, real sigma)
		{
			setParam(param_h, sigma);
		}

		virtual ~NonLocalMeanFilter() {}

	public:
		virtual void operator()(
			const vec3* src,
			uint32_t width, uint32_t height,
			vec3* dst) override final;

		void setParam(real param_h, real sigma)
		{
			m_param_h = param_h;
			m_sigma = sigma;
		}

	private:
		real m_param_h{ 0.2 };
		real m_sigma{ 0.2 };
	};
}
