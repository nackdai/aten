#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pathtracing.h"

namespace idaten
{
	class PathTracingGeometryRendering : public PathTracing {
	public:
		PathTracingGeometryRendering() {}
		virtual ~PathTracingGeometryRendering() {}

	public:
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

		struct AOV {
			float depth;
			int meshid;
			int mtrlid;
			float3 normal;
		};

	protected:
		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml) override;

		void renderAOVs(
			int width, int height,
			int sample, int maxSamples,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void getRenderAOVSize(int& w, int& h)
		{
			w <<= 1;
			h <<= 1;
		}

		virtual idaten::TypedCudaMemory<AOV>& getCurAOVs()
		{
			return m_aovs[0];
		}

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height,
			int maxSamples) override;

	protected:
		idaten::TypedCudaMemory<AOV> m_aovs[2];
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
			const std::vector<aten::GeomParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
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

		virtual idaten::TypedCudaMemory<AOV>& getCurAOVs() override final
		{
			return m_aovs[m_curAOV];
		}

		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
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

		aten::mat4 m_mtx_V2C;		// View - Clip.
		aten::mat4 m_mtx_C2V;		// Clip - View.
		aten::mat4 m_mtxPrevV2C;	// View - Clip.

		idaten::TypedCudaMemory<aten::mat4> m_mtxs;

		bool m_isFirstRender{ true };

		idaten::TypedCudaMemory<int> m_hitboolsTemporal;
		idaten::TypedCudaMemory<int> m_hitidxTemporal;
	};
}
