#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pt_standard_impl.h"
#include "sampler/sampler.h"
#include "renderer/aov.h"

namespace idaten
{
    class PathTracingImplBase : public StandardPT {
    public:
        PathTracingImplBase() = default;
        virtual ~PathTracingImplBase() = default;

        PathTracingImplBase(const PathTracingImplBase&) = delete;
        PathTracingImplBase(PathTracingImplBase&&) = delete;
        PathTracingImplBase& operator=(const PathTracingImplBase&) = delete;
        PathTracingImplBase& operator=(PathTracingImplBase&&) = delete;

        bool isEnableProgressive() const
        {
            return m_enableProgressive;
        }
        void setEnableProgressive(bool b)
        {
            m_enableProgressive = b;
        }

    protected:
        virtual void onHitTest(
            int32_t width, int32_t height,
            int32_t bounce);

        virtual void missShade(
            int32_t width, int32_t height,
            int32_t bounce)
        {
            StandardPT::missShade(
                width, height,
                bounce,
                aov_.normal_depth(),
                aov_.albedo_meshid());
        }

        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce);

        void onShadeByShadowRay(
            int32_t width, int32_t height,
            int32_t bounce);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t maxSamples);

    protected:
        // AOV buffer
        using AOVHostBuffer = AT_NAME::AOVHostBuffer<idaten::TypedCudaMemory<float4>, AT_NAME::AOVBufferType::NumBasicAovBuffer>;
        AOVHostBuffer aov_;

        bool m_enableProgressive{ false };
    };

    class PathTracing : public PathTracingImplBase {
    public:
        enum class Mode {
            PT,
            AOVar,  // Arbitrary Output Variables.
        };

        PathTracing() = default;
        virtual ~PathTracing() = default;

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

        void updateMaterial(const std::vector<aten::MaterialParameter>& mtrls);
        void updateLight(const std::vector<aten::LightParameter>& lights);

    protected:
        void onRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf);

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        Mode m_mode{ Mode::PT };

        // To export to GL.
        idaten::TypedCudaMemory<cudaSurfaceObject_t> gl_surface_cuda_rscs_;
        std::vector<idaten::CudaGLSurface> gl_surfaces_;
    };
}
