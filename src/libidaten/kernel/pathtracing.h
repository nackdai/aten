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

		virtual void update(
			GLuint gltex,
			int width, int height,
			const aten::CameraParameter& camera,
			const std::vector<aten::ShapeParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<std::vector<aten::BVHNode>>& nodes,
			const std::vector<aten::PrimitiveParamter>& prims,
			const std::vector<aten::vertex>& vtxs,
			const std::vector<aten::mat4>& mtxs,
			const std::vector<TextureResource>& texs,
			const EnvmapResource& envmapRsc) override final;

	private:
		inline void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			int seed);

		inline void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos);

		inline void onShadeMiss(
			int width, int height,
			int depth);

		inline void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int depth, int rrDepth,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		inline void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height);

	private:
		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;
	};
}
