#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"

namespace idaten
{
	class Renderer {
	protected:
		Renderer() {}
		~Renderer() {}

	public:
		virtual void render(
			aten::vec4* image,
			int width, int height) = 0;

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
			int envmapIdx);

		void updateCamera(const aten::CameraParameter& camera);

	protected:
		idaten::CudaMemory dst;
		idaten::TypedCudaMemory<aten::CameraParameter> cam;
		idaten::TypedCudaMemory<aten::ShapeParameter> shapeparam;
		idaten::TypedCudaMemory<aten::MaterialParameter> mtrlparam;
		idaten::TypedCudaMemory<aten::LightParameter> lightparam;
		idaten::TypedCudaMemory<aten::PrimitiveParamter> primparams;

		idaten::TypedCudaMemory<aten::mat4> mtxparams;
		
		std::vector<idaten::CudaTextureResource> nodeparam;
		idaten::TypedCudaMemory<cudaTextureObject_t> nodetex;

		std::vector<idaten::CudaTexture> texRsc;
		idaten::TypedCudaMemory<cudaTextureObject_t> tex;
		int m_envmapIdx{ -1 };

		idaten::CudaGLSurface glimg;
		idaten::CudaTextureResource vtxparamsPos;
		idaten::CudaTextureResource vtxparamsNml;
	};
}
