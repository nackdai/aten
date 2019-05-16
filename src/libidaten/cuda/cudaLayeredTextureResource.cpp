#include <memory.h>

#include "cuda/cudaTextureResource.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten
{
    void CudaLeyered2DTexture::init(
        std::vector<const aten::vec4*>& p,
        uint32_t width,
        uint32_t height)
    {
        int layerNum = static_cast<int>(p.size());
        int imgSize = width * height;

        auto totalSize = imgSize * layerNum;
        std::vector<aten::vec4> hostMem(totalSize);

        for (int n = 0; n < layerNum; n++) {
            for (int i = 0; i < imgSize; i++) {
                hostMem[n * imgSize + i] = p[n][i];
            }
        }

        auto bytes = imgSize * layerNum * sizeof(float4);

        // allocate device memory for result.
        float4* deviceMem = nullptr;
        checkCudaErrors(cudaMalloc((void **)&deviceMem, bytes));

        // allocate array and copy image data.
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        checkCudaErrors(
            cudaMalloc3DArray(
                &m_array,
                &channelDesc, 
                make_cudaExtent(width, height, layerNum),
                cudaArrayLayered));

        cudaMemcpy3DParms memcpyParams = { 0 };
        memcpyParams.srcPos = make_cudaPos(0, 0, 0);
        memcpyParams.dstPos = make_cudaPos(0, 0, 0);
        memcpyParams.srcPtr = make_cudaPitchedPtr(&hostMem[0], width * sizeof(float4), width, height);
        memcpyParams.dstArray = m_array;
        memcpyParams.extent = make_cudaExtent(width, height, layerNum);
        memcpyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3DAsync(&memcpyParams));

        memset(&m_resDesc, 0, sizeof(m_resDesc));
        m_resDesc.resType = cudaResourceTypeArray;
        m_resDesc.res.array.array = m_array;
    }

    cudaTextureObject_t CudaLeyered2DTexture::bind()
    {
        if (m_tex == 0) {
            // Make texture description:
            cudaTextureDesc tex_desc = {};
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.filterMode = cudaFilterModePoint;
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.normalizedCoords = true;

            checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
        }

        return m_tex;
    }
}