#include <memory.h>

#include "cuda/cudaTextureResource.h"

#include "kernel/GenMipMaps.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten
{
    CudaTextureResource::~CudaTextureResource()
    {
        unbind();
        if (m_buffer) {
            checkCudaErrors(cudaFree(m_buffer));
            m_buffer = nullptr;
        }
    }
    // NOTE
    // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

    void CudaTextureResource::init(
        const aten::vec4* p,
        uint32_t memberNumInItem,
        uint32_t numOfContaints)
    {
        onInit(p, memberNumInItem, numOfContaints);
    }

    void CudaTextureResource::onInit(
        const aten::vec4* p,
        uint32_t memberNumInItem,
        uint32_t numOfContaints)
    {
        auto size = sizeof(float4) * memberNumInItem * numOfContaints;

        if (m_buffer && m_size != size) {
            unbind();
            checkCudaErrors(cudaFree(m_buffer));
            m_buffer = nullptr;
        }

        m_size = size;

        if (!m_buffer) {
            checkCudaErrors(cudaMalloc(&m_buffer, m_size));
        }

        if (p) {
            checkCudaErrors(cudaMemcpyAsync(m_buffer, p, m_size, cudaMemcpyDefault));
        }

        // Make Resource description:
        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypeLinear;
        m_resDesc.res.linear.devPtr = m_buffer;
        m_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        m_resDesc.res.linear.desc.x = 32; // bits per channel
        m_resDesc.res.linear.desc.y = 32; // bits per channel
        m_resDesc.res.linear.desc.z = 32; // bits per channel
        m_resDesc.res.linear.desc.w = 32; // bits per channel
        m_resDesc.res.linear.sizeInBytes = memberNumInItem * numOfContaints * sizeof(float4);
    }

    cudaTextureObject_t CudaTextureResource::bind()
    {
        if (m_tex == 0) {
            // TODO
            // Only for resource array.

            // Make texture description:
            cudaTextureDesc tex_desc = {};
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.filterMode = cudaFilterModePoint;
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.normalizedCoords = 0;

            checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
        }

        return m_tex;
    }

    void CudaTextureResource::unbind()
    {
        if (m_tex > 0) {
            checkCudaErrors(cudaDestroyTextureObject(m_tex));
            m_tex = 0;
        }
    }

    void CudaTextureResource::update(
        const aten::vec4* p,
        uint32_t memberNumInItem,
        uint32_t numOfContaints,
        uint32_t offsetCount/*= 0*/)
    {
        AT_ASSERT(m_buffer);

        auto size = sizeof(float4) * memberNumInItem * numOfContaints;

        float4* dst = reinterpret_cast<float4*>(m_buffer);

        checkCudaErrors(cudaMemcpyAsync(dst + offsetCount, p, size, cudaMemcpyDefault));
    }

    void CudaTextureResource::read(void* p, uint32_t bytes)
    {
        AT_ASSERT(m_buffer);
        checkCudaErrors(cudaMemcpy(p, m_buffer, bytes, cudaMemcpyDefault));
    }

    /////////////////////////////////////////////////////

    void CudaTexture::init(
        const aten::vec4* p,
        int32_t width, int32_t height)
    {
        // NOTE
        // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%A5%EA%A5%CB%A5%A2%A5%E1%A5%E2%A5%EA%A4%C8CUDA%C7%DB%CE%F3
        // http://www.orangeowlsolutions.com/archives/613
        // Textures & Surfaces CUDA Webinar
        // http://developer.download.nvidia.com/CUDA/training/texture_webinar_aug_2011.pdf

#if 0
        // NOTE
        // 2Dテクスチャの場合は、pitchのアラインメントを考慮しないといけない.
        // cudaMallocPitch はアラインメントを考慮した処理になっている.

        size_t dstPitch = 0;
        size_t srcPitch = sizeof(float4) * width;

        checkCudaErrors(cudaMallocPitch(&m_buffer, &dstPitch, srcPitch, height));
        checkCudaErrors(cudaMemcpy2DAsync(m_buffer, dstPitch, p, srcPitch, srcPitch, height, cudaMemcpyHostToDevice));

        // Make Resource description:
        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypePitch2D;
        m_resDesc.res.pitch2D.devPtr = m_buffer;
        m_resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        m_resDesc.res.pitch2D.desc.x = 32; // bits per channel
        m_resDesc.res.pitch2D.desc.y = 32; // bits per channel
        m_resDesc.res.pitch2D.desc.z = 32; // bits per channel
        m_resDesc.res.pitch2D.desc.w = 32; // bits per channel
        m_resDesc.res.pitch2D.width = width;
        m_resDesc.res.pitch2D.height = height;
        m_resDesc.res.pitch2D.pitchInBytes = dstPitch;
#else
        m_channelFmtDesc.f = cudaChannelFormatKindFloat;
        m_channelFmtDesc.x = 32;
        m_channelFmtDesc.y = 32;
        m_channelFmtDesc.z = 32;
        m_channelFmtDesc.w = 32;

        checkCudaErrors(cudaMallocArray(&m_array, &m_channelFmtDesc, width, height));

#if 0
        // NOTE
        // cudaMemcpyToArrayAsync is deprecated
        size_t size = width * height * sizeof(float4);
        checkCudaErrors(cudaMemcpyToArrayAsync(m_array, 0, 0, p, size, cudaMemcpyHostToDevice));
#else
        auto pitch = width * sizeof(float4);
        checkCudaErrors(
            cudaMemcpy2DToArrayAsync(m_array, 0, 0, p, pitch, width, height, cudaMemcpyHostToDevice));
#endif

        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypeArray;
        m_resDesc.res.array.array = m_array;
#endif
    }

    cudaTextureObject_t CudaTexture::bind()
    {
        if (m_tex == 0) {
            if (m_isMipmap) {
                cudaTextureDesc tex_desc = {};

                tex_desc.normalizedCoords = 1;
                tex_desc.filterMode = cudaFilterModeLinear;
                tex_desc.mipmapFilterMode = cudaFilterModeLinear;

#if 0
                tex_desc.addressMode[0] = cudaAddressModeClamp;
                tex_desc.addressMode[1] = cudaAddressModeClamp;
                tex_desc.addressMode[2] = cudaAddressModeClamp;
#else
                tex_desc.addressMode[0] = cudaAddressModeWrap;
                tex_desc.addressMode[1] = cudaAddressModeWrap;
                tex_desc.addressMode[2] = cudaAddressModeWrap;
#endif

                tex_desc.maxMipmapLevelClamp = float(m_mipmapLevel - 1);

                tex_desc.readMode = cudaReadModeElementType;

                checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
            }
            else {
                // TODO
                // Only for resource array.

                // Make texture description:
                cudaTextureDesc tex_desc = {};
                tex_desc.readMode = cudaReadModeElementType;
                tex_desc.filterMode = cudaFilterModeLinear;
                tex_desc.addressMode[0] = cudaAddressModeWrap;
                tex_desc.addressMode[1] = cudaAddressModeWrap;
                tex_desc.normalizedCoords = 1;

                checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
            }
        }

        return m_tex;
    }

    ////////////////////////////////////////////////////////////////////////////////////

    static inline int32_t getMipMapLevels(int32_t width, int32_t height)
    {
        int32_t sz = std::max(width, height);

        int32_t levels = 0;

        while (sz)
        {
            sz /= 2;
            levels++;
        }

        return levels;
    }

    void CudaTexture::initAsMipmap(
        const aten::vec4* p,
        int32_t width, int32_t height,
        int32_t level)
    {
        level = std::min(level, getMipMapLevels(width, height));

        cudaExtent size;
        {
            size.width = width;
            size.height = height;
            size.depth = 0;
        }

        cudaMipmappedArray_t mipmapArray;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, level));

        // upload level 0.
        cudaArray_t level0;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

        void* data = const_cast<void*>((const void*)p);

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr(data, size.width * sizeof(float4), size.width, size.height);
        copyParams.dstArray = level0;
        copyParams.extent = size;
        copyParams.extent.depth = 1;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3DAsync(&copyParams));

        // compute rest of mipmaps based on level 0.
        generateMipMaps(mipmapArray, width, height, level);

        // Make Resource description:
        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypeMipmappedArray;
        m_resDesc.res.mipmap.mipmap = mipmapArray;

        m_isMipmap = true;
        m_mipmapLevel = level;
    }
}
