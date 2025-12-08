#pragma once

#include "aten4idaten.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"
#include "kernel/StreamCompaction.h"
#include "renderer/pathtracing/pt_params.h"

namespace aten {
    class Grid;
}

namespace idaten
{
    struct DeviceContextInHost;

    class Renderer {
    protected:
        Renderer();
        virtual ~Renderer() = default;

        Renderer(const Renderer&) = delete;
        Renderer(Renderer&&) = delete;
        Renderer& operator=(const Renderer&) = delete;
        Renderer& operator=(Renderer&&) = delete;

    public:
        enum class AOV {
            Albedo,
            Normal,
            WireFrame,
            BaryCentric,
        };

        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) = 0;

        virtual void reset()
        {
            m_frame = 1;
        }

        void updateBVH(
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            uint32_t triOffsetCount);

        void updateCamera(const aten::CameraParameter& camera);

        void UpdateTexture(int32_t idx, const aten::context& ctxt);

        void UpdateSceneRenderingConfig(const aten::context& ctxt);

        void viewTextures(
            uint32_t idx,
            int32_t screenWidth, int32_t screenHeight);

        void ViewAOV(
            AOV aov,
            int32_t screenWidth, int32_t screenHeight);

        uint32_t getRegisteredTextureNum() const;

        std::vector<idaten::CudaTextureResource>& getCudaTextureResourceForBvhNodes();

        idaten::CudaTextureResource getCudaTextureResourceForVtxPos();

        idaten::StreamCompaction& getCompaction();

        uint32_t GetFrameCount() const
        {
            return m_frame;
        }

        void setHitDistanceLimit(float d)
        {
            m_hitDistLimit = d;
        }

        aten::tuple<aten::ray, aten::vec3> getDebugInfo(uint32_t x, uint32_t y);

        cudaStream_t GetCudaStream() const
        {
            return m_stream;
        }

    protected:
        virtual void UpdateSceneData(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            uint32_t advance_prim_num,
            uint32_t advance_vtx_num,
            std::function<const aten::Grid*(const aten::context&)> proxy_get_grid_from_host_scene_context = nullptr);

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

        virtual void MissShadeWithFillingAov(
            int32_t width, int32_t height,
            int32_t bounce,
            idaten::TypedCudaMemory<float4>& aovNormalDepth,
            idaten::TypedCudaMemory<float4>& aovTexclrMeshid);

        std::shared_ptr<idaten::DeviceContextInHost> ctxt_host_;

        idaten::CudaGLSurface m_glimg;

        idaten::StreamCompaction m_compaction;

        aten::CameraParameter m_cam;

        uint32_t m_frame{ 1 };

        cudaStream_t m_stream{ (cudaStream_t)0 };

        std::shared_ptr<AT_NAME::PathHost> path_host_;

        idaten::TypedCudaMemory<aten::Intersection> m_isects;
        idaten::TypedCudaMemory<aten::ray> m_rays;

        idaten::TypedCudaMemory<int32_t> m_hitbools;
        idaten::TypedCudaMemory<int32_t> m_hitidx;

        idaten::TypedCudaMemory<aten::ShadowRay> m_shadowRays;

        // Distance limitation to kill path.
        float m_hitDistLimit{ AT_MATH_INF };

        idaten::TypedCudaMemory<uint32_t> m_sobolMatrices;
        idaten::TypedCudaMemory<uint32_t> m_random;
    };
}
