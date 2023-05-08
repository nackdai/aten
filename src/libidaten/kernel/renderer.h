#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"
#include "kernel/StreamCompaction.h"

namespace idaten
{
    struct TileDomain {
        int32_t x;
        int32_t y;
        int32_t w;
        int32_t h;

        TileDomain() {}
        TileDomain(int32_t _x, int32_t _y, int32_t _w, int32_t _h)
            : x(_x), y(_y), w(_w), h(_h)
        {}
    };

    struct EnvmapResource {
        int32_t idx{ -1 };
        real avgIllum;
        real multiplyer{ real(1) };

        EnvmapResource() {}

        EnvmapResource(int32_t i, real illum, real mul = real(1))
            : idx(i), avgIllum(illum), multiplyer(mul)
        {}
    };

    class Renderer {
    protected:
        Renderer() {}
        virtual ~Renderer() {}

    public:
        virtual void render(
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) = 0;

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
            const EnvmapResource& envmapRsc);

        virtual void reset() {}

        void updateBVH(
            const std::vector<aten::ObjectParameter>& geoms,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::mat4>& mtxs);

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
            return static_cast<uint32_t>(m_texRsc.size());
        }

        std::vector<idaten::CudaTextureResource>& getCudaTextureResourceForBvhNodes()
        {
            return m_nodeparam;
        }

        idaten::CudaTextureResource getCudaTextureResourceForVtxPos()
        {
            return m_vtxparamsPos;
        }

        idaten::StreamCompaction& getCompaction()
        {
            return m_compaction;
        }

    protected:
        idaten::StreamCompaction m_compaction;

        aten::CameraParameter m_cam;
        idaten::TypedCudaMemory<aten::ObjectParameter> m_shapeparam;
        idaten::TypedCudaMemory<aten::MaterialParameter> m_mtrlparam;
        idaten::TypedCudaMemory<aten::LightParameter> m_lightparam;
        idaten::TypedCudaMemory<aten::TriangleParameter> m_primparams;

        idaten::TypedCudaMemory<aten::mat4> m_mtxparams;

        std::vector<idaten::CudaTextureResource> m_nodeparam;
        idaten::TypedCudaMemory<cudaTextureObject_t> m_nodetex;

        std::vector<idaten::CudaTexture> m_texRsc;
        idaten::TypedCudaMemory<cudaTextureObject_t> m_tex;
        EnvmapResource m_envmapRsc;

        idaten::CudaGLSurface m_glimg;
        idaten::CudaTextureResource m_vtxparamsPos;
        idaten::CudaTextureResource m_vtxparamsNml;
    };
}
