#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class SVGFPathTracing : public Renderer {
	public:
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

		struct AOV {
			float depth;
			int meshid;
			int mtrlid;
			int padding0;

			float4 normal;
			float4 texclr;
			float4 color;
			float4 moments;
			float4 var;
		};
#else
		struct Path;
		struct AOV;
#endif

	public:
		SVGFPathTracing() {}
		virtual ~SVGFPathTracing() {}

	public:
		virtual void render(
			aten::vec4* image,
			int width, int height,
			int maxSamples,
			int maxBounce) override final;

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
			const EnvmapResource& envmapRsc) override;

	protected:
		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			int seed,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos);

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

		void onTemporalReprojection(
			cudaSurfaceObject_t outputSurf,
			int width, int height);

		void onVarianceEstimation(
			cudaSurfaceObject_t outputSurf,
			int width, int height);

		void onAtrousFilter(
			cudaSurfaceObject_t outputSurf,
			int width, int height);

		idaten::TypedCudaMemory<AOV>& getCurAovs()
		{
			return m_aovs[m_curAOVPos];
		}
		idaten::TypedCudaMemory<AOV>& getPrevAovs()
		{
			return m_aovs[1 - m_curAOVPos];
		}

	protected:
		idaten::TypedCudaMemory<Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;

		int m_curAOVPos{ 0 };

		idaten::TypedCudaMemory<AOV> m_aovs[2];

		aten::mat4 m_mtxV2C;		// View - Clip.
		aten::mat4 m_mtxC2V;		// Clip - View.
		aten::mat4 m_mtxPrevV2C;	// View - Clip.

		idaten::TypedCudaMemory<aten::mat4> m_mtxs;

		bool m_isFirstRender{ true };

		idaten::TypedCudaMemory<float4> m_atrousClr[2];
		idaten::TypedCudaMemory<float4> m_atrousVar[2];
	};
}
