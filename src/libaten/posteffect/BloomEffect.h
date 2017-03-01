#pragma once

#include "visualizer/MultiPassPostProc.h"

namespace aten {
	class BloomEffect : public MultiPassPostProc {
	public:
		BloomEffect() {}
		virtual ~BloomEffect() {}

	public:
		bool init(
			int width, int height,
			PixelFormat inFmt, PixelFormat outFmt,
			const char* pathVS,
			const char* pathFS_4x4,
			const char* pathFS_2x2, 
			const char* pathFS_HBlur,
			const char* pathFS_VBlur,
			const char* pathFS_GaussBlur,
			const char* pathFS_Final);

		virtual PixelFormat inFormat() const override final
		{
			return m_fmtIn;
		}

		virtual PixelFormat outFormat() const override final
		{
			return m_fmtOut;
		}

		virtual uint32_t getOutWidth() const override final
		{
			return m_passFinal.getFbo().getWidth();
		}
		virtual uint32_t getOutHeight() const override final
		{
			return m_passFinal.getFbo().getHeight();
		}

		virtual FBO& getFbo() override final
		{
			return m_passFinal.getFbo();
		}

	private:
		class BloomEffectPass : public visualizer::PostProc {
		public:
			BloomEffectPass() {}
			virtual ~BloomEffectPass() {}

		public:
			bool init(
				int srcWidth, int srcHeight,
				int dstWidth, int dstHeight,
				PixelFormat inFmt, PixelFormat outFmt,
				const char* pathVS,
				const char* pathFS);

			virtual void prepareRender(
				const void* pixels,
				bool revert) override;

			virtual PixelFormat inFormat() const override
			{
				return m_fmtIn;
			}
			virtual PixelFormat outFormat() const override
			{
				return m_fmtOut;
			}

		protected:
			int m_srcWidth;
			int m_srcHeight;

			PixelFormat m_fmtIn;
			PixelFormat m_fmtOut;
		};

		class BloomEffectFinalPass : public BloomEffectPass {
		public:
			BloomEffectFinalPass() {}
			virtual ~BloomEffectFinalPass() {}

			virtual void prepareRender(
				const void* pixels,
				bool revert) override;

			BloomEffect* m_body;
		};

		PixelFormat m_fmtIn;
		PixelFormat m_fmtOut;

		BloomEffectPass m_pass4x4;
		BloomEffectPass m_pass2x2;
		BloomEffectPass m_passHBlur;
		BloomEffectPass m_passVBlur;
		BloomEffectPass m_passGaussBlur_4x4;
		BloomEffectPass m_passGaussBlur_2x2;
		BloomEffectFinalPass m_passFinal;
	};
}