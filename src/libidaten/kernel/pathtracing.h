#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class PathTracing : public Renderer {
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
		PathTracing() {}
		virtual ~PathTracing() {}

	public:
		void prepare();

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

		virtual void enableRenderAOV(
			GLuint gltexPosition,
			GLuint gltexNormal,
			GLuint gltexAlbedo,
			aten::vec3& posRange) override;

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

	protected:
		idaten::TypedCudaMemory<idaten::PathTracing::Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;
		idaten::TypedCudaMemory<idaten::PathTracing::ShadowRay> m_shadowRays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;

		bool m_enableAOV{ false };
		aten::vec3 m_posRange{ aten::vec3(real(1)) };
		idaten::TypedCudaMemory<cudaSurfaceObject_t> m_aovCudaRsc;
		std::vector<idaten::CudaGLSurface> m_aovs;
	};

	class PathTracingGeometryRendering : public PathTracing {
	public:
		PathTracingGeometryRendering() {}
		virtual ~PathTracingGeometryRendering() {}

	public:
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
			cudaTextureObject_t texVtxNml) override;

		void renderAOVs(
			int width, int height,
			int sample, int maxSamples,
			int seed,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void getRenderAOVSize(int& w, int& h)
		{
			w <<= 1;
			h <<= 1;
		}

		virtual idaten::TypedCudaMemory<float4>& getCurAOVs()
		{
			return m_aovs[0];
		}

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height,
			int maxSamples) override;

	protected:
		idaten::TypedCudaMemory<float4> m_aovs[2];
	};

	class PathTracingTemporalReprojection : public PathTracingGeometryRendering {
	public:
		PathTracingTemporalReprojection() {}
		virtual ~PathTracingTemporalReprojection() {}

	public:
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

		virtual void reset() override final
		{
			//m_isFirstRender = true;
		}

	protected:
		virtual void getRenderAOVSize(int& w, int& h) override final
		{
			w = w;
			h = h;
		}

		virtual idaten::TypedCudaMemory<float4>& getCurAOVs() override final
		{
			return m_aovs[m_curAOV];
		}

		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			int seed,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml) override final;

#if 1
		virtual void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos) override final;

		virtual void onShadeMiss(
			int width, int height,
			int bounce) override final;

		virtual void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int width, int height,
			int bounce, int rrBounce,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml) override final;
#endif

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height,
			int maxSamples) override final;

	protected:
		int m_curAOV{ 0 };

		aten::mat4 m_mtxV2C;		// View - Clip.
		aten::mat4 m_mtxC2V;		// Clip - View.
		aten::mat4 m_mtxPrevV2C;	// View - Clip.

		idaten::TypedCudaMemory<aten::mat4> m_mtxs;

		bool m_isFirstRender{ true };

		idaten::TypedCudaMemory<int> m_hitboolsTemporal;
		idaten::TypedCudaMemory<int> m_hitidxTemporal;
	};
}
