#pragma once

#include "aten4idaten.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"
#include "kernel/StreamCompaction.h"

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

    public:
        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) = 0;

        virtual void reset() {}

        void updateBVH(
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            uint32_t triOffsetCount);

        void updateCamera(const aten::CameraParameter& camera);

        void UpdateSceneRenderingConfig(const aten::context& ctxt);

        void viewTextures(
            uint32_t idx,
            int32_t screenWidth, int32_t screenHeight);

        uint32_t getRegisteredTextureNum() const;

        std::vector<idaten::CudaTextureResource>& getCudaTextureResourceForBvhNodes();

        idaten::CudaTextureResource getCudaTextureResourceForVtxPos();

        idaten::StreamCompaction& getCompaction();

    protected:
        virtual void UpdateSceneData(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            uint32_t advance_prim_num,
            uint32_t advance_vtx_num,
            const aten::BackgroundResource& bg_resource,
            std::function<const aten::Grid*(const aten::context&)> proxy_get_grid_from_host_scene_context = nullptr);

        std::shared_ptr<idaten::DeviceContextInHost> ctxt_host_;

        idaten::CudaGLSurface m_glimg;

        idaten::StreamCompaction m_compaction;

        aten::CameraParameter m_cam;
        aten::BackgroundResource bg_;
    };
}
