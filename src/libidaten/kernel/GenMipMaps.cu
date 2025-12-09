#include <algorithm>

#include "kernel/GenMipMaps.h"

#include "cuda/helper_math.h"
#include "cuda/cudamemory.h"
#include "cuda/cudautil.h"

// NOTE
// http://www.cse.uaa.alaska.edu/~ssiewert/a490dmis_code/CUDA/cuda_work/samples/2_Graphics/bindlessTexture/bindlessTexture_kernel.cu

__global__ void genMipmap(
    cudaSurfaceObject_t mipOutput,
    cudaTextureObject_t mipInput,
    int32_t imageW, int32_t imageH)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float(imageW);
    float py = 1.0 / float(imageH);

    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));


        color /= 4.0f;

        surf2Dwrite(color, mipOutput, x * sizeof(float4), y);
    }
}

namespace idaten {
    void generateMipMaps(
        cudaMipmappedArray_t mipmapArray,
        int32_t width, int32_t height,
        int32_t maxLevel)
    {
        int32_t level = 0;

        //while (width != 1 || height != 1)
        while (level + 1 < maxLevel)
        {
            width /= 2;
            height /= 2;

            width = std::max(1, width);
            height = std::max(1, height);

            // Copy from.
            cudaArray_t levelFrom;
            checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));

            // Copy to.
            cudaArray_t levelTo;
            checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

            cudaExtent levelToSize;
            checkCudaErrors(cudaArrayGetInfo(nullptr, &levelToSize, nullptr, levelTo));
            AT_ASSERT(levelToSize.width == width);
            AT_ASSERT(levelToSize.height == height);
            AT_ASSERT(levelToSize.depth == 0);

            // generate texture object for reading
            cudaTextureObject_t texInput;
            {
                cudaResourceDesc texRes;
                {
                    memset(&texRes, 0, sizeof(cudaResourceDesc));

                    texRes.resType = cudaResourceTypeArray;
                    texRes.res.array.array = levelFrom;
                }

                cudaTextureDesc texDesc;
                {
                    memset(&texDesc, 0, sizeof(cudaTextureDesc));

                    // To use UV coordinates in [0,1] range, set normalizedCoords to 1.
                    texDesc.normalizedCoords = 1;
                    texDesc.filterMode = cudaFilterModeLinear;
                    texDesc.addressMode[0] = cudaAddressModeWrap;
                    texDesc.addressMode[1] = cudaAddressModeWrap;
                    texDesc.addressMode[2] = cudaAddressModeWrap;
                    texDesc.readMode = cudaReadModeElementType;
                }

                checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDesc, nullptr));
            }

            // generate surface object for writing
            cudaSurfaceObject_t surfOutput;
            {
                cudaResourceDesc surfRes;
                {
                    memset(&surfRes, 0, sizeof(cudaResourceDesc));
                    surfRes.resType = cudaResourceTypeArray;
                    surfRes.res.array.array = levelTo;
                }

                checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));
            }

            // run mipmap kernel
            dim3 block(16, 16, 1);
            dim3 grid(
                (width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y, 1);

            genMipmap << <grid, block >> > (
                surfOutput,
                texInput,
                width, height);

            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaGetLastError());

            checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

            checkCudaErrors(cudaDestroyTextureObject(texInput));

            level++;
        }
    }
}
