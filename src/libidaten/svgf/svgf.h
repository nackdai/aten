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
        SVGFPathTracing()
        {
            can_ssrt_hit_test_ = true;
        }
        virtual ~SVGFPathTracing() {}

    public:
        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) override;

        void UpdateSceneData(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            uint32_t advance_prim_num,
            uint32_t advance_vtx_num,
            const aten::BackgroundResource& bg_resource,
            std::function<const aten::Grid* (const aten::context&)> proxy_get_grid_from_host_scene_context = nullptr) override;

        void SetGBuffer(
            GLuint gltexGbuffer,
            GLuint gltexMotionDepthbuffer);

        Mode GetMode() const
        {
            return m_mode;
        }
        void SetMode(Mode mode)
        {
            auto prev = m_mode;
            m_mode = mode;
            if (prev != m_mode) {
                reset();
            }
        }

        AT_NAME::SVGFAovMode GetAOVMode() const
        {
            return m_aovMode;
        }
        void SetAOVMode(AT_NAME::SVGFAovMode mode)
        {
            m_aovMode = mode;
        }

        virtual void reset() override final
        {
            StandardPT::reset();
            params_.curr_aov_pos = 0;
        }

        void WillPickPixel(int32_t ix, int32_t iy)
        {
            m_willPicklPixel = true;
            m_pickedInfo.ix = ix;
            m_pickedInfo.iy = iy;
        }

        bool GetPickedPixelInfo(PickedInfo& ret)
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

        uint32_t getAtrousIterCount() const
        {
            return params_.atrous_iter_cnt;
        }
        void setAtrousIterCount(uint32_t c)
        {
            params_.atrous_iter_cnt = aten::clamp(c, 0U, 5U);
        }

    protected:
        void OnRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf);

        void onDenoise(
            int32_t width, int32_t height,
            cudaSurfaceObject_t outputSurf);

        void missShade(
            int32_t width, int32_t height,
            int32_t bounce) override
        {
            auto& curaov = params_.GetCurrAovBuffer();

            StandardPT::MissShadeWithFillingAov(
                width, height,
                bounce,
                curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>(),
                curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>());
        }

        void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce, int32_t max_depth) override;

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

        void OnDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void pick(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height);

        bool IsFirstFrame() const
        {
            return (m_frame == 1);
        }

        void SetStream(cudaStream_t stream);

    protected:
        AT_NAME::SVGFParams<idaten::TypedCudaMemory<float4>, idaten::CudaGLSurface> params_;

        idaten::TypedCudaMemory<PickedInfo> m_pick;

        bool m_willPicklPixel{ false };
        PickedInfo m_pickedInfo;

        Mode m_mode{ Mode::SVGF };
        AT_NAME::SVGFAovMode m_aovMode{ AT_NAME::SVGFAovMode::WireFrame };

        float m_depthThresholdTF{ 0.05f };
        float m_nmlThresholdTF{ 0.98f };

        bool m_isListedTextureObject{ false };
    };
}
