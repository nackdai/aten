#pragma once

#include "defs.h"
#include "math/vec3.h"
#include "visualizer/shader.h"

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
				const vec3* src,
				uint32_t width, uint32_t height,
				vec3* dst) = 0;
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
		};

	public:
		static bool init(int width, int height, PixelFormat fmt);

		static void addPreProc(PreProc* preproc);

		static void addPostProc(PostProc* postproc);

		static void setShader(shader* shader);

		static void render(
			const vec3* pixels,
			bool revert);
	};
}