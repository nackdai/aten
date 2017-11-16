#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class SSRT : public Renderer {
	public:
		struct ShadowRay : public aten::ray {
			aten::vec3 lightcontrib;
			real distToLight;
			int targetLightId;

			struct {
				uint32_t isActive : 1;
			};
		};

#ifdef __AT_CUDA__
		struct Path {
			aten::vec3 throughput;
			aten::vec3 contrib;
			aten::sampler sampler;

			real pdfb;
			int samples;

			bool isHit;
			bool isTerminate;
			bool isSingular;
			bool isKill;
		};
		C_ASSERT((sizeof(Path) % 4) == 0);
#else
		struct Path;
#endif

	public:
		SSRT() {}
		virtual ~SSRT() {}

	public:
		void prepare();

		virtual void render(
			int width, int height,
			int maxSamples,
			int maxBounce) override final;

		virtual void update(
			GLuint gltex,
			int width, int height,
			const aten::CameraParameter& camera,
			const std::vector<aten::GeomParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
			const std::vector<aten::PrimitiveParamter>& prims,
			const std::vector<aten::vertex>& vtxs,
			const std::vector<aten::mat4>& mtxs,
			const std::vector<TextureResource>& texs,
			const EnvmapResource& envmapRsc) override;

		void setGBuffer(
			GLuint gltexGbuffer,
			GLuint gltexDepth);

	protected:
		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onHitTest(
			int width, int height,
			int bounce,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onShadeMiss(
			int width, int height,
			int bounce);

		virtual void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int width, int height,
			int bounce, int rrBounce,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height,
			int maxSamples);

	protected:
		idaten::TypedCudaMemory<idaten::SSRT::Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;
		idaten::TypedCudaMemory<idaten::SSRT::ShadowRay> m_shadowRays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<int> m_notIntersectInScreenSpaceBools;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
		idaten::TypedCudaMemory<unsigned int> m_random;

		idaten::CudaGLSurface m_gbuffer;
		idaten::CudaGLSurface m_depth;

		uint32_t m_frame{ 1 };
	};
}
