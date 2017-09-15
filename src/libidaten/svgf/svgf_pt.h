#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class SVGFPathTracing : public Renderer {
	public:
		enum Mode {
			SVGF,			// Spatio-temporal Variance Guided Filter.
			TF,				// Temporal Filter.
			PT,				// Path Tracing.
			VAR,			// Variance (For debug).
			AOVar,			// Arbitrary Output Variables.
		};

		enum AOVMode {
			Normal,
			TexColor,
			Depth,
			WireFrame,
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

		struct AOV {
			float depth;
			int meshid;
			int mtrlid;
			float temporalWeight;

			float3 normal;
			float var;

			float4 texclr;
			float4 color;

			float4 moments;
		};
#else
		struct Path;
		struct AOV;
#endif

		struct PickedInfo {
			int ix{ -1 };
			int iy{ -1 };
			aten::vec3 color;
			aten::vec3 normal;
			float depth;
			int meshid;
			int triid;
			int mtrlid;
		};

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

		Mode getMode() const
		{
			return m_mode;
		}
		void setMode(Mode mode)
		{
			auto prev = m_mode;
			m_mode = mode;
			if (prev != m_mode) {
				reset();
			}
		}

		AOVMode getAOVMode() const
		{
			return m_aovMode;
		}
		void setAOVMode(AOVMode mode)
		{
			m_aovMode = mode;
		}

		virtual void reset() override final
		{
			m_frame = 1;
			m_curAOVPos = 0;
		}

		uint32_t frame() const
		{
			return m_frame;
		}

		void willPickPixel(int ix, int iy)
		{
			m_willPicklPixel = true;
			m_pickedInfo.ix = ix;
			m_pickedInfo.iy = iy;
		}

		bool getPickedPixelInfo(PickedInfo& ret)
		{
			bool isValid = (m_pickedInfo.ix >= 0);

			ret = m_pickedInfo;

			m_pickedInfo.ix = -1;
			m_pickedInfo.iy = -1;

			return isValid;
		}

		void setTemporalWeightThreshold(float th)
		{
			m_thresholdTemporalWeight = aten::clamp(th, 0.0f, 1.0f);
		}

		float getTemporalWeightThreshold() const
		{
			return m_thresholdTemporalWeight;
		}

		void setAtrousTapRadiusScale(int s)
		{
			m_atrousTapRadiusScale = std::max(s, 1);
		}

		int getAtrousTapRadiusScale() const
		{
			return m_atrousTapRadiusScale;
		}

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

		void copyFromTmpBufferToAov(int width, int height);

		void onFillAOV(
			cudaSurfaceObject_t outputSurf,
			int width, int height);

		void pick(
			int ix, int iy,
			int width, int height,
			cudaTextureObject_t texVtxPos);

		idaten::TypedCudaMemory<AOV>& getCurAovs()
		{
			return m_aovs[m_curAOVPos];
		}
		idaten::TypedCudaMemory<AOV>& getPrevAovs()
		{
			return m_aovs[1 - m_curAOVPos];
		}

		bool isFirstFrame() const
		{
			return (m_frame == 1);
		}

	protected:
		idaten::TypedCudaMemory<Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
		idaten::TypedCudaMemory<unsigned int> m_random;

		int m_curAOVPos{ 0 };

		idaten::TypedCudaMemory<AOV> m_aovs[2];

		aten::mat4 m_mtxV2C;		// View - Clip.
		aten::mat4 m_mtxC2V;		// Clip - View.
		aten::mat4 m_mtxPrevV2C;	// View - Clip.

		idaten::TypedCudaMemory<aten::mat4> m_mtxs;

		unsigned int m_frame{ 1 };

		idaten::TypedCudaMemory<float4> m_atrousClr[2];
		idaten::TypedCudaMemory<float> m_atrousVar[2];

		idaten::TypedCudaMemory<float4> m_tmpBuf;

		float m_thresholdTemporalWeight{ 0.0f };
		int m_atrousTapRadiusScale{ 1 };

		idaten::TypedCudaMemory<PickedInfo> m_pick;

		bool m_willPicklPixel{ false };
		PickedInfo m_pickedInfo;

		Mode m_mode{ Mode::SVGF };
		AOVMode m_aovMode{ AOVMode::WireFrame };
	};
}
