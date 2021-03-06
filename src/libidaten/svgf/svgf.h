#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"
#include "sampler/sampler.h"

namespace idaten
{
    class SVGFPathTracing : public Renderer {
    public:
        enum Mode {
            SVGF,   // Spatio-temporal Variance Guided Filter.
            TF,     // Temporal Filter.
            PT,     // Path Tracing.
            VAR,    // Variance (For debug).
            AOVar,  // Arbitrary Output Variables.
        };

        enum AOVMode {
            Normal,
            TexColor,
            Depth,
            WireFrame,
            BaryCentric,
            Motion,
            ObjId,
        };

        static const int ShadowRayNum = 2;

#ifdef __AT_CUDA__
        struct PathThroughput {
            aten::vec3 throughput;
            real pdfb;
        };

        struct PathContrib {
            union {
                float4 v;
                struct {
                    float3 contrib;
                    float samples;
                };
            };
        };

        struct PathAttribute {
            union {
                float4 v;
                struct {
                    bool isHit;
                    bool isTerminate;
                    bool isSingular;
                    bool isKill;

                    aten::MaterialType mtrlType;
                };
            };
        };

        struct Path {
            PathThroughput* throughput;
            PathContrib* contrib;
            PathAttribute* attrib;
            aten::sampler* sampler;
        };

        struct ShadowRay {
            aten::vec3 rayorg;

            aten::vec3 raydir;
            aten::vec3 lightcontrib;
            real distToLight;

            uint8_t targetLightId;

            struct {
                uint32_t isActive : 1;
            };
        };
#else
        struct Path;
        struct PathThroughput;
        struct PathContrib;
        struct PathAttribute;
        struct ShadowRay;
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
            const TileDomain& tileDomain,
            int maxSamples,
            int maxBounce) override;

        virtual void update(
            GLuint gltex,
            int width, int height,
            const aten::CameraParameter& camera,
            const std::vector<aten::GeomParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::PrimitiveParamter>& prims,
            uint32_t advancePrimNum,
            const std::vector<aten::vertex>& vtxs,
            uint32_t advanceVtxNum,
            const std::vector<aten::mat4>& mtxs,
            const std::vector<TextureResource>& texs,
            const EnvmapResource& envmapRsc) override;

        void updateBVH(
            const std::vector<aten::GeomParameter>& geoms,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::mat4>& mtxs);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::PrimitiveParamter>& triangles,
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

        void setHitDistanceLimit(float d)
        {
            m_hitDistLimit = d;
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
        void onInit(int width, int height);

        void onClear();

        void onRender(
            const TileDomain& tileDomain,
            int width, int height,
            int maxSamples,
            int maxBounce,
            cudaSurfaceObject_t outputSurf,
            cudaTextureObject_t vtxTexPos,
            cudaTextureObject_t vtxTexNml);

        virtual void onDenoise(
            const TileDomain& tileDomain,
            int width, int height,
            cudaSurfaceObject_t outputSurf);

        virtual void onGenPath(
            int sample, int maxBounce,
            int seed,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onHitTest(
            int width, int height,
            int bounce,
            cudaTextureObject_t texVtxPos);

        virtual void onScreenSpaceHitTest(
            int width, int height,
            int bounce,
            cudaTextureObject_t texVtxPos);

        virtual void onShadeMiss(
            int width, int height,
            int bounce,
            int offsetX = -1,
            int offsetY = -1);

        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            int sample,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        void onShadeByShadowRay(
            int bounce,
            cudaTextureObject_t texVtxPos);

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

        void onAtrousFilterIter(
            uint32_t iterCnt,
            uint32_t maxIterCnt,
            cudaSurfaceObject_t outputSurf,
            int width, int height);

        void onCopyFromTmpBufferToAov(int width, int height);

        void onDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            cudaTextureObject_t texVtxPos);

        void onCopyBufferForTile(int width, int height);

        void pick(
            int ix, int iy,
            int width, int height,
            cudaTextureObject_t texVtxPos);

        int getCurAovs()
        {
            return m_curAOVPos;
        }
        int getPrevAovs()
        {
            return 1 - m_curAOVPos;
        }

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        bool m_isInitPash{ false };
        idaten::TypedCudaMemory<Path> m_paths;
        idaten::TypedCudaMemory<PathThroughput> m_pathThroughput;
        idaten::TypedCudaMemory<PathContrib> m_pathContrib;
        idaten::TypedCudaMemory<PathAttribute> m_pathAttrib;
        idaten::TypedCudaMemory<aten::sampler> m_pathSampler;

        idaten::TypedCudaMemory<aten::Intersection> m_isects;
        idaten::TypedCudaMemory<aten::ray> m_rays;

        idaten::TypedCudaMemory<int> m_hitbools;
        idaten::TypedCudaMemory<int> m_hitidx;

        idaten::TypedCudaMemory<ShadowRay> m_shadowRays;

        idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
        idaten::TypedCudaMemory<unsigned int> m_random;

        // Current AOV buffer position.
        int m_curAOVPos{ 0 };

        // AOV buffer. Current frame and previous frame.
        idaten::TypedCudaMemory<float4> m_aovNormalDepth[2];
        idaten::TypedCudaMemory<float4> m_aovTexclrMeshid[2];
        idaten::TypedCudaMemory<float4> m_aovColorVariance[2];
        idaten::TypedCudaMemory<float4> m_aovMomentTemporalWeight[2];

        aten::mat4 m_mtxW2V;        // World - View.
        aten::mat4 m_mtxV2C;        // View - Clip.
        aten::mat4 m_mtxC2V;        // Clip - View.

        // View - World.
        aten::mat4 m_mtxV2W;
        aten::mat4 m_mtxPrevW2V;

        uint32_t m_frame{ 1 };

        // For A-trous wavelet.
        idaten::TypedCudaMemory<float4> m_atrousClrVar[2];

        idaten::TypedCudaMemory<float4> m_tmpBuf;

        // Distance limitation to kill path.
        real m_hitDistLimit{ AT_MATH_INF };

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

        TileDomain m_tileDomain;

        bool m_isListedTextureObject{ false };
        bool m_canSSRTHitTest{ true };

        cudaStream_t m_stream{ (cudaStream_t)0 };
    };
}
