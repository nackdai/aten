#pragma once

#include "cuda/cudamemory.h"

#include "kernel/pt_params.h"
#include "kernel/renderer.h"

namespace idaten {
    class StandardPT : public Renderer {
    public:
        StandardPT() = default;
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

        void setEnableEnvmap(bool b)
        {
            m_enableEnvmap = b;
        }

        aten::tuple<aten::ray, aten::vec3> getDebugInfo(uint32_t x, uint32_t y);

    protected:
        virtual void initSamplerParameter(
            int width, int height)
        {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
            m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
            m_sobolMatrices.writeFromHostToDeviceByNum(sobol::Matrices::matrices, m_sobolMatrices.num());
#endif

            auto& r = aten::getRandom();
            m_random.init(width * height);
            m_random.writeFromHostToDeviceByNum(&r[0], width * height);
        }

        virtual bool initPath(
            int width, int height);

        virtual void clearPath();

        virtual void generatePath(
            bool needFillAOV,
            int sample, int maxBounce,
            int seed,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void hitTest(
            int width, int height,
            int bounce,
            cudaTextureObject_t texVtxPos);

        virtual void hitTestOnScreenSpace(
            int width, int height,
            idaten::CudaGLSurface& gbuffer,
            cudaTextureObject_t texVtxPos);

        virtual void missShade(
            int width, int height,
            int bounce,
            idaten::TypedCudaMemory<float4>& aovNormalDepth,
            idaten::TypedCudaMemory<float4>& aovTexclrMeshid,
            int offsetX = -1,
            int offsetY = -1);

    protected:
        uint32_t m_frame{ 1 };

        TileDomain m_tileDomain;

        cudaStream_t m_stream{ (cudaStream_t)0 };

        bool m_isInitPath{ false };
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

        // Distance limitation to kill path.
        real m_hitDistLimit{ AT_MATH_INF };

        idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
        idaten::TypedCudaMemory<unsigned int> m_random;

        bool m_enableEnvmap{ true };
    };
}
