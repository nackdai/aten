#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"

namespace idaten
{
	class RayTracing {
	public:
		RayTracing() {}
		~RayTracing() {}

	public:
		void prepare();

		void update(
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
		aten::CudaMemory dst;
		aten::TypedCudaMemory<aten::CameraParameter> cam;
		aten::TypedCudaMemory<aten::ShapeParameter> shapeparam;
		aten::TypedCudaMemory<aten::MaterialParameter> mtrlparam;
		aten::TypedCudaMemory<aten::LightParameter> lightparam;
		aten::TypedCudaMemory<aten::BVHNode> nodeparam;
		aten::TypedCudaMemory<aten::PrimitiveParamter> primparams;
		aten::TypedCudaMemory<aten::vertex> vtxparams;
	};
}
