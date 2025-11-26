#pragma once

#include <vector>

#include "math/vec4.h"
#include "image/texture.h"
#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten
{
    class CudaTextureResource {
    public:
        CudaTextureResource() = default;
        virtual ~CudaTextureResource();

    public:
        void init(
            const aten::vec4* p,
            size_t memberNumInItem,
            size_t numOfContaints);

        cudaTextureObject_t bind();
        void unbind();

        void update(
            const aten::vec4* p,
            size_t memberNumInItem,
            size_t numOfContaints,
            int32_t offsetCount = 0);

        void read(void* p, size_t bytes);

        size_t size() const
        {
            return m_size;
        }

    private:
        void onInit(
            const aten::vec4* p,
            size_t memberNumInItem,
            size_t numOfContaints);

    protected:
        void* m_buffer{ nullptr };
        size_t m_size{ 0 };
        cudaResourceDesc m_resDesc;
        cudaTextureObject_t m_tex{ 0 };
    };

    class CudaTexture {
    public:
        CudaTexture() = default;
        ~CudaTexture() = default;

    public:
        void init(
            const aten::vec4* p,
            int32_t width, int32_t height,
            aten::TextureFilterMode filter = aten::TextureFilterMode::Linear,
            aten::TextureAddressMode address = aten::TextureAddressMode::Wrap);

        void initAsMipmap(
            const aten::vec4* p,
            int32_t width, int32_t height,
            int32_t level,
            aten::TextureFilterMode filter = aten::TextureFilterMode::Linear,
            aten::TextureAddressMode address = aten::TextureAddressMode::Wrap);

        cudaTextureObject_t bind();
        void unbind();

        void CopyFromHost(
            const aten::vec4* p,
            int32_t width, int32_t height);

    private:
        static inline cudaTextureFilterMode ConvertFilterMode(aten::TextureFilterMode filter);

        static inline cudaTextureAddressMode ConvertAddressMode(aten::TextureAddressMode address);

        bool m_isMipmap{ false };
        int32_t m_mipmapLevel{ 0 };

        cudaArray_t m_array{ nullptr };
        cudaMipmappedArray_t mipmapped_array_{ nullptr };
        cudaChannelFormatDesc m_channelFmtDesc;

        cudaResourceDesc m_resDesc;
        cudaTextureObject_t m_tex{ 0 };

        cudaTextureFilterMode filter_mode_{ cudaTextureFilterMode::cudaFilterModePoint };
        cudaTextureAddressMode address_mode_{ cudaTextureAddressMode::cudaAddressModeWrap };
    };
}
