#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"

namespace idaten
{
	struct EnvmapResource {
		int idx{ -1 };
		real avgIllum;
		real multiplyer{ real(1) };

		EnvmapResource() {}

		EnvmapResource(int i, real illum, real mul = real(1))
			: idx(i), avgIllum(illum), multiplyer(mul)
		{}
	};

	class Renderer {
	protected:
		Renderer() {}
		~Renderer() {}

	public:
		virtual void render(
			aten::vec4* image,
			int width, int height,
			int maxSamples,
			int maxBounce) = 0;

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
			const EnvmapResource& envmapRsc);

		virtual void reset() {}

		void updateCamera(const aten::CameraParameter& camera);

		virtual void enableRenderAOV(
			GLuint gltexPosition,
			GLuint gltexNormal,
			GLuint gltexAlbedo,
			aten::vec3& posRange)
		{
			// Nothing is done...
		}

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
		EnvmapResource m_envmapRsc;

		idaten::CudaGLSurface glimg;
		idaten::CudaTextureResource vtxparamsPos;
		idaten::CudaTextureResource vtxparamsNml;

		aten::CameraParameter m_camParam;
	};
}
