#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "texture/texture.h"

namespace aten {
	class ATrousDenoiser : public MultiPassPostProc {
	public:
		ATrousDenoiser() {}
		~ATrousDenoiser() {}

	public:
		bool init(
			int width, int height,
			const char* vsPath,
			const char* fsPath);

		texture& getNormalBuffer()
		{
			return m_normal;
		}
		texture& getPositionBuffer()
		{
			return m_pos;
		}

		virtual PixelFormat inFormat() const override final
		{
			return PixelFormat::rgba32f;
		}

		virtual PixelFormat outFormat() const override final
		{
			return PixelFormat::rgba32f;
		}

		virtual FBO& getFbo() override final
		{
			return m_pass[ITER - 1].getFbo();
		}

	private:
		class ATrousPass : public visualizer::PostProc {
		public:
			ATrousPass() {}
			virtual ~ATrousPass() {}

		public:
			virtual void prepareRender(
				const void* pixels,
				bool revert) override;

			virtual PixelFormat inFormat() const override final
			{
				return PixelFormat::rgba32f;
			}
			virtual PixelFormat outFormat() const override final
			{
				return PixelFormat::rgba32f;
			}

			ATrousDenoiser* m_body{ nullptr };
			int m_idx{ -1 };
		};

		static const int ITER = 5;

		texture m_pos;
		texture m_normal;

		ATrousPass m_pass[ITER];
	};
}