#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pathtracing.h"

namespace idaten
{
	class DirectLightRenderer : public PathTracing {
	public:
		DirectLightRenderer() {}
		virtual ~DirectLightRenderer() {}

	protected:
		virtual void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int width, int height,
			int bounce, int rrBounce,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);
	};
}
