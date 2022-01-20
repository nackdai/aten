#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pt_params.h"
#include "kernel/pt_standard_impl.h"
#include "sampler/sampler.h"

namespace idaten
{
    class PathTracing : public StandardPT {
    public:
        enum class Mode {
            PT,
            AOVar,  // Arbitrary Output Variables.
        };

        PathTracing() = default;
        virtual ~PathTracing() = default;

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

        virtual void reset() override final
        {
            m_frame = 1;
        }

        uint32_t frame() const
        {
            return m_frame;
        }

        void setHitDistanceLimit(float d)
        {
            m_hitDistLimit = d;
        }

        bool isEnableProgressive() const
        {
            return m_enableProgressive;
        }
        void setEnableProgressive(bool b)
        {
            m_enableProgressive = b;
        }

    protected:
        void onRender(
            const TileDomain& tileDomain,
            int width, int height,
            int maxSamples,
            int maxBounce,
            cudaSurfaceObject_t outputSurf,
            cudaTextureObject_t vtxTexPos,
            cudaTextureObject_t vtxTexNml);

        virtual void onHitTest(
            int width, int height,
            int bounce,
            cudaTextureObject_t texVtxPos);

        void missShade(
            int width, int height,
            int bounce,
            int offsetX = -1,
            int offsetY = -1)
        {
            StandardPT::missShade(
                width, height,
                bounce,
                m_aovNormalDepth,
                m_aovTexclrMeshid,
                offsetX, offsetY);
        }

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

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        Mode m_mode{ Mode::PT };

        // AOV buffer
        idaten::TypedCudaMemory<float4> m_aovNormalDepth;
        idaten::TypedCudaMemory<float4> m_aovTexclrMeshid;

        aten::mat4 m_mtxW2V;    // World - View.
        aten::mat4 m_mtxV2C;    // View - Clip.
        aten::mat4 m_mtxC2V;    // Clip - View.

        // View - World.
        aten::mat4 m_mtxV2W;
        aten::mat4 m_mtxPrevW2V;

        // G-Buffer rendered by OpenGL.
        idaten::CudaGLSurface m_gbuffer;
        idaten::CudaGLSurface m_motionDepthBuffer;

        bool m_isListedTextureObject{ false };

        bool m_enableProgressive{ false };
    };
}
