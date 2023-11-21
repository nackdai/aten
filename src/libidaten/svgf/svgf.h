#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pathtracing.h"
#include "sampler/sampler.h"

#include "renderer/svgf/svgf_types.h"

namespace idaten
{
    class SVGFPathTracing : public PathTracingImplBase {
    public:
        enum Mode {
            SVGF,   // Spatio-temporal Variance Guided Filter.
            TF,     // Temporal Filter.
            PT,     // Path Tracing.
            VAR,    // Variance (For debug).
            AOVar,  // Arbitrary Output Variables.
        };

        struct PickedInfo {
            int32_t ix{ -1 };
            int32_t iy{ -1 };
            aten::vec3 color;
            aten::vec3 normal;
            float depth;
            int32_t meshid;
            int32_t triid;
            int32_t mtrlid;
        };

    public:
        SVGFPathTracing() {}
        virtual ~SVGFPathTracing() {}

    public:
        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) override;

        virtual void update(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const std::vector<aten::ObjectParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::TriangleParameter>& prims,
            uint32_t advancePrimNum,
            const std::vector<aten::vertex>& vtxs,
            uint32_t advanceVtxNum,
            const std::vector<aten::mat4>& mtxs,
            const std::vector<TextureResource>& texs,
            const EnvmapResource& envmapRsc) override;

        void setGBuffer(
            GLuint gltexGbuffer,
            GLuint gltexMotionDepthbuffer);

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

        AT_NAME::SVGFAovMode getAOVMode() const
        {
            return m_aovMode;
        }
        void setAOVMode(AT_NAME::SVGFAovMode mode)
        {
            m_aovMode = mode;
        }

        virtual void reset() override final
        {
            StandardPT::reset();
            params_.curr_aov_pos = 0;
        }

        void willPickPixel(int32_t ix, int32_t iy)
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

        float getTemporalFilterDepthThreshold() const
        {
            return m_depthThresholdTF;
        }
        float getTemporalFilterNormalThreshold() const
        {
            return m_nmlThresholdTF;
        }

        void setTemporalFilterDepthThreshold(float th)
        {
            m_depthThresholdTF = aten::clamp(th, 0.01f, 1.0f);
        }
        void setTemporalFilterNormalThreshold(float th)
        {
            m_nmlThresholdTF = aten::clamp(th, 0.0f, 1.0f);
        }

        bool canSSRTHitTest() const
        {
            return m_canSSRTHitTest;
        }

        void setCanSSRTHitTest(bool f)
        {
            m_canSSRTHitTest = f;
        }

        uint32_t getAtrousIterCount() const
        {
            return params_.atrous_iter_cnt;
        }
        void setAtrousIterCount(uint32_t c)
        {
            params_.atrous_iter_cnt = aten::clamp(c, 0U, 5U);
        }

    protected:
        void onRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf);

        void onDenoise(
            int32_t width, int32_t height,
            cudaSurfaceObject_t outputSurf);

        void onHitTest(
            int32_t width, int32_t height,
            int32_t bounce)  override;

        void missShade(
            int32_t width, int32_t height,
            int32_t bounce,
            int32_t offsetX = -1,
            int32_t offsetY = -1)
        {
            auto& curaov = params_.GetCurrAovBuffer();

            StandardPT::missShade(
                width, height,
                bounce,
                curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>(),
                curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>());
        }

        void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce) override;

        void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t maxSamples) override;

        void onTemporalReprojection(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void onVarianceEstimation(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void onAtrousFilter(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void onAtrousFilterIter(
            uint32_t filter_iter_count,
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void onCopyFromTmpBufferToAov(int32_t width, int32_t height);

        void onDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void pick(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height);

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        AT_NAME::SVGFParams<idaten::TypedCudaMemory<float4>, idaten::CudaGLSurface> params_;

        // G-Buffer rendered by OpenGL.
        idaten::CudaGLSurface m_gbuffer;

        idaten::TypedCudaMemory<PickedInfo> m_pick;

        bool m_willPicklPixel{ false };
        PickedInfo m_pickedInfo;

        Mode m_mode{ Mode::SVGF };
        AT_NAME::SVGFAovMode m_aovMode{ AT_NAME::SVGFAovMode::WireFrame };

        float m_depthThresholdTF{ 0.05f };
        float m_nmlThresholdTF{ 0.98f };

        bool m_isListedTextureObject{ false };
        bool m_canSSRTHitTest{ true };
    };
}
