#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class PathTracing : public Renderer {
	public:
		PathTracing() {}
		~PathTracing() {}

	public:
		void prepare();

		virtual void render(
			aten::vec4* image,
			int width, int height) override final;
	};
}
