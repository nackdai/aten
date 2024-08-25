#pragma once

#include "aten4idaten.h"
#include "kernel/StreamCompaction.h"
#include "kernel/device_scene_context.cuh"

namespace idaten
{
    class Renderer {
    protected:
        Renderer() {}
        virtual ~Renderer() {}

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

        void viewTextures(
            uint32_t idx,
            int32_t screenWidth, int32_t screenHeight);

        uint32_t getRegisteredTextureNum() const
        {
            return static_cast<uint32_t>(ctxt_host_.texRsc.size());
        }

        std::vector<idaten::CudaTextureResource>& getCudaTextureResourceForBvhNodes()
        {
            return ctxt_host_.nodeparam;
        }

        idaten::CudaTextureResource getCudaTextureResourceForVtxPos()
        {
            return ctxt_host_.vtxparamsPos;
        }

        idaten::StreamCompaction& getCompaction()
        {
            return m_compaction;
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
            const aten::BackgroundResource& bg_resource);

        idaten::DeviceContextInHost ctxt_host_;

        idaten::CudaGLSurface m_glimg;

        idaten::StreamCompaction m_compaction;

        aten::CameraParameter m_cam;
        aten::BackgroundResource bg_;
    };
}
