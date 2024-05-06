#pragma once

#include "cuda/cudamemory.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/renderer.h"

namespace idaten {
    struct PathHost;
    struct ShadowRay;

    class StandardPT : public Renderer {
    public:
        StandardPT();
        virtual ~StandardPT() {}

        StandardPT(const StandardPT&) = delete;
        StandardPT(StandardPT&&) = delete;
        StandardPT& operator=(const StandardPT&) = delete;
        StandardPT& operator=(StandardPT&&) = delete;

        virtual void reset() override
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

        void SetEnableEnvmap(bool b)
        {
            m_enableEnvmap = b;
        }

        aten::tuple<aten::ray, aten::vec3> getDebugInfo(uint32_t x, uint32_t y);

    protected:
        virtual void initSamplerParameter(
            int32_t width, int32_t height)
        {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
            m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
            m_sobolMatrices.writeFromHostToDeviceByNum(sobol::Matrices::matrices, m_sobolMatrices.num());
#endif

            auto& r = aten::getRandom();
            m_random.resize(width * height);
            m_random.writeFromHostToDeviceByNum(&r[0], width * height);
        }

        virtual bool InitPath(
            int32_t width, int32_t height);

        virtual void clearPath();

        virtual void generatePath(
            int32_t width, int32_t height,
            bool needFillAOV,
            int32_t sample, int32_t maxBounce,
            int32_t seed);

        virtual void hitTest(
            int32_t width, int32_t height,
            int32_t bounce);

        virtual void hitTestOnScreenSpace(
            int32_t width, int32_t height,
            idaten::CudaGLSurface& gbuffer);

        virtual void missShade(
            int32_t width, int32_t height,
            int32_t bounce,
            idaten::TypedCudaMemory<float4>& aovNormalDepth,
            idaten::TypedCudaMemory<float4>& aovTexclrMeshid);

    protected:
        uint32_t m_frame{ 1 };

        cudaStream_t m_stream{ (cudaStream_t)0 };

        std::shared_ptr<PathHost> path_host_;

        idaten::TypedCudaMemory<aten::Intersection> m_isects;
        idaten::TypedCudaMemory<aten::ray> m_rays;

        idaten::TypedCudaMemory<int32_t> m_hitbools;
        idaten::TypedCudaMemory<int32_t> m_hitidx;

        idaten::TypedCudaMemory<ShadowRay> m_shadowRays;

        // Distance limitation to kill path.
        real m_hitDistLimit{ AT_MATH_INF };

        idaten::TypedCudaMemory<uint32_t> m_sobolMatrices;
        idaten::TypedCudaMemory<uint32_t> m_random;

        bool m_enableEnvmap{ true };
    };
}
