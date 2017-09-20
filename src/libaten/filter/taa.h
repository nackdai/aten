#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "texture/texture.h"
#include "camera/pinhole.h"
#include "math/mat4.h"

namespace aten {
	class TAA : public MultiPassPostProc {
	public:
		TAA() {}
		virtual ~TAA() {}

	public:
		bool init(
			int width, int height,
			const char* taaVsPath, const char* taaFsPath,
			const char* finalVsPath, const char* finalFsPath);

		void update(
			uint32_t frame,
			const PinholeCamera& cam);

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
			return m_final.getFbo();
		}

		uint32_t getAovGLTexHandle() const
		{
			return m_aovTex.getGLTexHandle();
		}

		void enableTAA(bool e)
		{
			m_enableTAA = e;
		}
		bool isEnableTAA() const
		{
			return m_enableTAA;
		}

		void showTAADiff(bool s)
		{
			m_canShowAADiff = s;
		}
		bool canShowTAADiff() const
		{
			return m_canShowAADiff;
		}

	private:
		void prepareFbo(const uint32_t* tex, int num, std::vector<uint32_t>& comps);

		class TAAPass : public visualizer::PostProc {
		public:
			TAAPass() {}
			virtual ~TAAPass() {}

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

			TAA* m_body{ nullptr };
		};

		class FinalPass : public visualizer::PostProc {
		public:
			FinalPass() {}
			virtual ~FinalPass() {}

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

			TAA* m_body{ nullptr };
		};

	private:
		TAAPass m_taa;
		FinalPass m_final;

		texture m_aovTex;

		aten::mat4 m_mtxW2V;		// World - View.
		aten::mat4 m_mtxV2C;		// View - Clip.
		aten::mat4 m_mtxC2V;		// Clip - View.

		// View - World.
		aten::mat4 m_mtxV2W;
		aten::mat4 m_mtxPrevW2V;

		int m_idx{ 0 };

		bool m_enableTAA{ true };
		bool m_canShowAADiff{ false };
	};
}
