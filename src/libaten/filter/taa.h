#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "camera/pinhole.h"

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

		int m_idx{ 0 };
	};
}
