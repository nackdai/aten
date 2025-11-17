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

        bool IsEnableProgressive() const
        {
            return m_enableProgressive;
        }
        void SetEnableProgressive(bool b)
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
            StandardPT::MissShadeWithFillingAov(
                width, height,
                bounce,
                aov_.normal_depth(),
                aov_.albedo_meshid());
        }

        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce, int32_t max_depth);

        void onShadeByShadowRay(
            int32_t width, int32_t height,
            int32_t bounce);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t maxSamples);

        virtual void DisplayAOV(
            cudaSurfaceObject_t output_surface,
            int32_t width, int32_t height);

        void SetGBuffer(GLuint gltexGbuffer);

        bool CanSSRTHitTest() const
        {
            return can_ssrt_hit_test_;
        }

        void SetCanSSRTHitTest(bool f)
        {
            can_ssrt_hit_test_ = f;
        }

    protected:
        // AOV buffer
        using AOVHostBuffer = AT_NAME::AOVHostBuffer<idaten::TypedCudaMemory<float4>, AT_NAME::AOVBufferType::NumBasicAovBuffer>;
        AOVHostBuffer aov_;

        // G-Buffer rendered by OpenGL.
        idaten::CudaGLSurface g_buffer_;

        bool m_enableProgressive{ false };
        bool can_ssrt_hit_test_{ false };
    };

    class PathTracing : public PathTracingImplBase {
    public:
        enum class Mode {
            PT,
            AOV,    // Arbitrary Output Variables.
        };

        PathTracing() = default;
        virtual ~PathTracing() = default;

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

        void updateMaterial(const std::vector<aten::MaterialParameter>& mtrls);
        void updateLight(const aten::context& scene_ctxt);

        void SetRenderingMode(Mode mode)
        {
            m_mode = mode;
        }
        Mode GetRenderingMode() const
        {
            return m_mode;
        }

    protected:
        virtual void OnRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf);

        bool IsFirstFrame() const
        {
            return (m_frame == 1);
        }

        void SetStream(cudaStream_t stream);

    protected:
        Mode m_mode{ Mode::PT };
    };
}
