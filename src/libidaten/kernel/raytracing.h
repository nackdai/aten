#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class RayTracing : public Renderer {
	public:
		RayTracing() {}
		~RayTracing() {}

	public:
		void prepare();

		virtual void render(
			int width, int height,
			int maxSamples,
			int maxBounce) override final;
	};
}
