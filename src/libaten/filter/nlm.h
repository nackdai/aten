#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

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
			const vec4* src,
			uint32_t width, uint32_t height,
			vec4* dst) override final;

		void setParam(real param_h, real sigma)
		{
			m_param_h = param_h;
			m_sigma = sigma;
		}

		virtual void setParam(Values& values) override final
		{
			m_param_h = values.get("h", m_param_h);
			m_sigma = values.get("sigma", m_sigma);
		}

	private:
		real m_param_h{ 0.2 };
		real m_sigma{ 0.2 };
	};

	class NonLocalMeanFilterShader : public Blitter {
	public:
		NonLocalMeanFilterShader() {}
		NonLocalMeanFilterShader(real param_h, real sigma)
		{
			setParam(param_h, sigma);
		}

		virtual ~NonLocalMeanFilterShader() {}

	public:
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;

		void setParam(real param_h, real sigma)
		{
			m_param_h = param_h;
			m_sigma = sigma;
		}

		virtual void setParam(Values& values) override final
		{
			m_param_h = values.get("h", m_param_h);
			m_sigma = values.get("sigma", m_sigma);
		}

	private:
		real m_param_h{ 0.2 };
		real m_sigma{ 0.2 };
	};

}
