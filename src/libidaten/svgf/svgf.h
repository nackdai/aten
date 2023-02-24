#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pt_params.h"
#include "kernel/pt_standard_impl.h"
#include "sampler/sampler.h"

namespace idaten
{
    class SVGFPathTracing : public StandardPT {
    public:
        enum Mode {
            SVGF,   // Spatio-temporal Variance Guided Filter.
            TF,     // Temporal Filter.
            PT,     // Path Tracing.
            VAR,    // Variance (For debug).
            AOVar,  // Arbitrary Output Variables.
        };

        struct AOVMode : public AT_NAME::AOVType {
            enum Type {
                ObjId = AT_NAME::AOVType::MeshId,
                TexColor = AT_NAME::AOVType::Albedo,
                WireFrame = AT_NAME::AOVType::BeginOfInheritType,
                BaryCentric,
                Motion,
                end_of_AOVMode = Motion,
            };

            static constexpr size_t Num = static_cast<size_t>(Type::end_of_AOVMode) + 1;

            AT_DEVICE_API AOVMode() : AT_NAME::AOVType(AT_NAME::AOVType::Normal) {}
            AT_DEVICE_API ~AOVMode() = default;
            AT_DEVICE_API AOVMode(int32_t type) : AT_NAME::AOVType(static_cast<Type>(type)) {}
        };

        static const int32_t ShadowRayNum = 2;

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
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) override;

        virtual void update(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const std::vector<aten::GeometryParameter>& shapes,
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

        void updateBVH(
            const std::vector<aten::GeometryParameter>& geoms,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::mat4>& mtxs);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            uint32_t triOffsetCount);

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
            StandardPT::reset();
            m_curAOVPos = 0;
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
            return m_atrousMaxIterCnt;
        }
        void setAtrousIterCount(uint32_t c)
        {
            m_atrousMaxIterCnt = aten::clamp(c, 0U, 5U);
        }

    protected:
        void onRender(
            const TileDomain& tileDomain,
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf,
            cudaTextureObject_t vtxTexPos,
            cudaTextureObject_t vtxTexNml);

        virtual void onDenoise(
            const TileDomain& tileDomain,
            int32_t width, int32_t height,
            cudaSurfaceObject_t outputSurf);

        virtual void onHitTest(
            int32_t width, int32_t height,
            int32_t bounce,
            cudaTextureObject_t texVtxPos);

        void missShade(
            int32_t width, int32_t height,
            int32_t bounce,
            int32_t offsetX = -1,
            int32_t offsetY = -1)
        {
            int32_t curaov_idx = getCurAovs();
            auto& curaov = aov_[curaov_idx];

            StandardPT::missShade(
                width, height,
                bounce,
                curaov.get<AOVBuffer::NormalDepth>(),
                curaov.get<AOVBuffer::AlbedoMeshId>(),
                offsetX, offsetY);
        }

        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        void onShadeByShadowRay(
            int32_t bounce,
            cudaTextureObject_t texVtxPos);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t maxSamples);

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
            uint32_t iterCnt,
            uint32_t maxIterCnt,
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void onCopyFromTmpBufferToAov(int32_t width, int32_t height);

        void onDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            cudaTextureObject_t texVtxPos);

        void onCopyBufferForTile(int32_t width, int32_t height);

        void pick(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            cudaTextureObject_t texVtxPos);

        int32_t getCurAovs()
        {
            return m_curAOVPos;
        }
        int32_t getPrevAovs()
        {
            return 1 - m_curAOVPos;
        }

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        // Current AOV buffer position.
        int32_t m_curAOVPos{ 0 };

        struct AOVBuffer : public AT_NAME::AOVBufferType {
            enum Type {
                ColorVariance = AT_NAME::AOVBufferType::BeginOfInheritType,
                MomentTemporalWeight,
                end_of_AOVBuffer = MomentTemporalWeight,
            };

            static constexpr size_t Num = static_cast<size_t>(Type::end_of_AOVBuffer) + 1;

            AT_DEVICE_API AOVBuffer() = default;
            AT_DEVICE_API ~AOVBuffer() = default;
            AT_DEVICE_API AOVBuffer(int32_t type) : AT_NAME::AOVBufferType(static_cast<Type>(type)) {}
        };

        using AOVHostBuffer = AT_NAME::AOVHostBuffer<idaten::TypedCudaMemory<float4>, AOVBuffer::Num>;
        std::array<AOVHostBuffer, 2> aov_;  // AOV buffer. Current frame and previous frame.

        aten::mat4 m_mtxW2V;        // World - View.
        aten::mat4 m_mtxV2C;        // View - Clip.
        aten::mat4 m_mtxC2V;        // Clip - View.

        // View - World.
        aten::mat4 m_mtxV2W;
        aten::mat4 m_mtxPrevW2V;

        // For A-trous wavelet.
        idaten::TypedCudaMemory<float4> m_atrousClrVar[2];

        idaten::TypedCudaMemory<float4> m_tmpBuf;

        // G-Buffer rendered by OpenGL.
        idaten::CudaGLSurface m_gbuffer;
        idaten::CudaGLSurface m_motionDepthBuffer;

        uint32_t m_atrousMaxIterCnt{ 5 };

        idaten::TypedCudaMemory<PickedInfo> m_pick;

        bool m_willPicklPixel{ false };
        PickedInfo m_pickedInfo;

        Mode m_mode{ Mode::SVGF };
        AOVMode m_aovMode{ AOVMode::WireFrame };

        float m_depthThresholdTF{ 0.05f };
        float m_nmlThresholdTF{ 0.98f };

        bool m_isListedTextureObject{ false };
        bool m_canSSRTHitTest{ true };
    };
}
