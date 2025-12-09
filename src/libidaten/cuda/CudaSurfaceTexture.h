#pragma once

#include "defs.h"
#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten
{
    template <class T>
    class CudaSurfaceTexture {
    public:
        using ValueType = T;

        CudaSurfaceTexture() = default;
        ~CudaSurfaceTexture()
        {
            Destroy();
        }

        CudaSurfaceTexture(const CudaSurfaceTexture&) = delete;
        CudaSurfaceTexture(CudaSurfaceTexture&& other) = delete;
        CudaSurfaceTexture& operator=(const CudaSurfaceTexture&) = delete;
        CudaSurfaceTexture& operator=(CudaSurfaceTexture&& other) = delete;

        bool Init(int32_t width, int32_t height)
        {
            if (surf_obj_ != 0) {
                return false;
            }

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
            auto result = checkCudaErrors(cudaMallocArray(
                &cuda_array_,
                &channelDesc,
                width, height,
                cudaArraySurfaceLoadStore));
            if (!result) {
                AT_ASSERT(false);
                return false;
            }

            result = CreateSurfaceObject();
            if (!result) {
                AT_ASSERT(false);
                Destroy();
                return false;
            }

            result = CreateTextureObject();
            if (!result) {
                AT_ASSERT(false);
                Destroy();
                return false;
            }

            return true;
        }

        void Destroy()
        {
            if (surf_obj_ != 0) {
                checkCudaErrors(cudaDestroySurfaceObject(surf_obj_));
                surf_obj_ = 0;
            }
            if (tex_obj_ != 0) {
                checkCudaErrors(cudaDestroyTextureObject(tex_obj_));
                tex_obj_ = 0;
            }
            if (cuda_array_) {
                checkCudaErrors(cudaFreeArray(cuda_array_));
                cuda_array_ = nullptr;
            }
        }

        cudaSurfaceObject_t GetSurfaceObject() const
        {
            return surf_obj_;
        }

        cudaTextureObject_t GetTextureObject() const
        {
            return tex_obj_;
        }

    private:
        bool CreateSurfaceObject()
        {
            cudaResourceDesc surf_rsc_desc;
            memset(&surf_rsc_desc, 0, sizeof(surf_rsc_desc));
            surf_rsc_desc.resType = cudaResourceTypeArray;
            surf_rsc_desc.res.array.array = cuda_array_;

            surf_obj_ = 0;
            auto result = checkCudaErrors(cudaCreateSurfaceObject(&surf_obj_, &surf_rsc_desc));

            return result;
        }

        bool CreateTextureObject()
        {
            cudaResourceDesc tex_rsc_desc;
            memset(&tex_rsc_desc, 0, sizeof(tex_rsc_desc));
            tex_rsc_desc.resType = cudaResourceTypeArray;
            tex_rsc_desc.res.array.array = cuda_array_;

            cudaTextureDesc tex_desc;
            memset(&tex_desc, 0, sizeof(tex_desc));

            // To use UV coordinates in [0,1] range, set normalizedCoords to 1.
            tex_desc.normalizedCoords = 1;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.addressMode[0] = cudaAddressModeClamp;
            tex_desc.addressMode[1] = cudaAddressModeClamp;
            tex_desc.readMode = cudaReadModeElementType;

            tex_obj_ = 0;
            auto result = checkCudaErrors(cudaCreateTextureObject(&tex_obj_, &tex_rsc_desc, &tex_desc, nullptr));

            return result;
        }

    private:
        cudaArray_t cuda_array_{ nullptr };
        cudaSurfaceObject_t surf_obj_{ 0 };
        cudaTextureObject_t tex_obj_{ 0 };
    };
}
