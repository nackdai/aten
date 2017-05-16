#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"

namespace idaten
{
	class RayTracing {
	public:
		RayTracing() {}
		~RayTracing() {}

	public:
		void prepare();

		void update(
			GLuint gltex,
			int width, int height,
			const aten::CameraParameter& camera,
			const std::vector<aten::ShapeParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<aten::BVHNode>& nodes,
			const std::vector<aten::PrimitiveParamter>& prims,
			const std::vector<aten::vertex>& vtxs);

		void render(
			aten::vec4* image,
			int width, int height);

	private:
		idaten::CudaMemory dst;
		idaten::TypedCudaMemory<aten::CameraParameter> cam;
		idaten::TypedCudaMemory<aten::ShapeParameter> shapeparam;
		idaten::TypedCudaMemory<aten::MaterialParameter> mtrlparam;
		idaten::TypedCudaMemory<aten::LightParameter> lightparam;
		idaten::TypedCudaMemory<aten::BVHNode> nodeparam;
		idaten::TypedCudaMemory<aten::PrimitiveParamter> primparams;

		idaten::CudaGLSurface glimg;
		idaten::CudaTextureResource vtxparams;
	};
}
