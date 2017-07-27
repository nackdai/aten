#pragma once

#include "defs.h"
#include "math/vec4.h"
#include "visualizer/pixelformat.h"
#include "visualizer/shader.h"
#include "visualizer/fbo.h"
#include "misc/value.h"

namespace aten {
	class visualizer {
	private:
		visualizer() {}
		~visualizer() {}

	public:
		class PreProc {
		protected:
			PreProc() {}
			virtual ~PreProc() {}

		public:
			virtual void operator()(
				const vec4* src,
				uint32_t width, uint32_t height,
				vec4* dst) = 0;

			virtual void setParam(Values& values) {}
		};

		class PostProc : public shader {
		protected:
			PostProc() {}
			virtual ~PostProc() {}

		public:
			virtual void prepareRender(
				const void* pixels,
				bool revert) override
			{
				shader::prepareRender(pixels, revert);
			}

			virtual PixelFormat inFormat() const = 0;
			virtual PixelFormat outFormat() const = 0;

			virtual void setParam(Values& values) {}

			virtual uint32_t getOutWidth() const
			{
				return m_width;
			}
			virtual uint32_t getOutHeight() const
			{
				return m_height;
			}

			virtual FBO& getFbo()
			{
				return m_fbo;
			}
			const FBO& getFbo() const
			{
				return m_fbo;
			}

			PostProc* getPrevPass()
			{
				return m_prevPass;
			}

			void prepareRender(
				PostProc* prevPass,
				const void* pixels,
				bool revert)
			{
				m_prevPass = prevPass;
				prepareRender(pixels, revert);
			}

		private:
			FBO m_fbo;
			PostProc* m_prevPass{ nullptr };
		};

	public:
		static uint32_t getTexHandle();
		static PixelFormat getPixelFormat();

		static bool init(int width, int height);

		static void addPreProc(PreProc* preproc);

		static bool addPostProc(PostProc* postproc);

		static void render(
			const vec4* pixels,
			bool revert);

		static void render(bool revert);
		static void render(uint32_t gltex, bool revert);

		static void clear();

		static void takeScreenshot(const char* filename);
	};
}