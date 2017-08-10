#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class SVGFPathTracing : public Renderer {
	public:
		enum AOVType {
			normal,
			depth_meshid,
			texclr,
			lum,
			clr_history,
			lum_histroy,

			num,
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

	protected:
		idaten::TypedCudaMemory<Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;

		int m_curAOVPos{ 0 };

		struct AOVs {
			aten::texture tex[AOVType::num];
		} m_aovTex[2];

		idaten::TypedCudaMemory<cudaSurfaceObject_t> m_aovCudaRsc[2];
		std::vector<idaten::CudaGLSurface> m_aovs[2];
	};
}
