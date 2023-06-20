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

        size_t size() const
        {
            return m_size;
        }

    private:
        void onInit(
            const aten::vec4* p,
            uint32_t memberNumInItem,
            uint32_t numOfContaints);

    protected:
        void* m_buffer{ nullptr };
        size_t m_size{ 0 };
        cudaResourceDesc m_resDesc;
        cudaTextureObject_t m_tex{ 0 };
    };

    struct TextureResource {
        const aten::vec4* ptr;
        int32_t width;
        int32_t height;

        TextureResource(const aten::vec4* p, int32_t w, int32_t h)
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
            int32_t width, int32_t height);

        void initAsMipmap(
            const aten::vec4* p,
            int32_t width, int32_t height,
            int32_t level);

        virtual cudaTextureObject_t bind() override final;

    private:
        bool m_isMipmap{ false };
        int32_t m_mipmapLevel{ 0 };

        cudaArray_t m_array{ nullptr };
        cudaChannelFormatDesc m_channelFmtDesc;
    };
}
