#pragma once

#include <vector>

#include "math/vec4.h"
#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten
{
    class CudaTextureResource {
    public:
        CudaTextureResource() {}
        virtual ~CudaTextureResource() {}

    public:
        void init(
            const aten::vec4* p,
            uint32_t memberNumInItem, 
            uint32_t numOfContaints);

        virtual cudaTextureObject_t bind();
        void unbind();

        void update(
            const aten::vec4* p,
            uint32_t memberNumInItem,
            uint32_t numOfContaints,
            uint32_t offsetCount = 0);

        void read(void* p, uint32_t bytes);

    private:
        void onInit(
            const aten::vec4* p,
            uint32_t memberNumInItem,
            uint32_t numOfContaints);

    protected:
        void* m_buffer{ nullptr };
        cudaResourceDesc m_resDesc;
        cudaTextureObject_t m_tex{ 0 };
    };

    struct TextureResource {
        const aten::vec4* ptr;
        int width;
        int height;

        TextureResource(const aten::vec4* p, int w, int h)
            : ptr(p), width(w), height(h)
        {}
    };

    class CudaTexture : public CudaTextureResource {
    public:
        CudaTexture() {}
        virtual ~CudaTexture() final {}

    public:
        void init(
            const aten::vec4* p,
            int width, int height);

        void initAsMipmap(
            const aten::vec4* p,
            int width, int height,
            int level);

        virtual cudaTextureObject_t bind() override final;

    private:
        bool m_isMipmap{ false };
        int m_mipmapLevel{ 0 };

        cudaArray_t m_array{ nullptr };
        cudaChannelFormatDesc m_channelFmtDesc;
    };

    class CudaLeyered2DTexture : public CudaTextureResource {
    public:
        CudaLeyered2DTexture() {}
        virtual ~CudaLeyered2DTexture() final {}

    public:
        void init(
            std::vector<const aten::vec4*>& p,
            uint32_t width,
            uint32_t height);

        virtual cudaTextureObject_t bind() override final;

        int getWidth() const
        {
            return m_width;
        }

        int getHeight() const
        {
            return m_height;
        }

        int getLayerNum() const
        {
            return m_layerNum;
        }

    private:
        cudaArray_t m_array{ nullptr };
        int m_width{ 0 };
        int m_height{ 0 };
        int m_layerNum{ 0 };
    };
}