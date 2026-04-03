#pragma once

#include <type_traits>

#include "defs.h"
#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

#include "image/texture.h"

namespace idaten
{
    struct SurfaceTexture
    {
        cudaSurfaceObject_t surface{ 0 };
        cudaTextureObject_t texture{ 0 };
    };

    template <class T>
    class CudaSurfaceTexture {
    public:
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, float2> || std::is_same_v<T, float4>
            || std::is_same_v<T, int> || std::is_same_v<T, int2> || std::is_same_v < T, int4>,
            "");
        
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

        bool Init(
            int32_t width, int32_t height,
            aten::TextureFilterMode filter = aten::TextureFilterMode::Point)
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

            result = CreateTextureObject(filter);
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

        SurfaceTexture GetSurfaceTexture() const
        {
            return SurfaceTexture{ surf_obj_, tex_obj_ };
        }

    protected:
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

        bool CreateTextureObject(aten::TextureFilterMode filter)
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
            tex_desc.addressMode[2] = cudaAddressModeClamp;
            tex_desc.filterMode = filter == aten::TextureFilterMode::Linear
                ? cudaFilterModeLinear
                : cudaFilterModePoint;
            tex_desc.readMode = cudaReadModeElementType;

            tex_obj_ = 0;
            auto result = checkCudaErrors(cudaCreateTextureObject(&tex_obj_, &tex_rsc_desc, &tex_desc, nullptr));

            return result;
        }

    protected:
        cudaArray_t cuda_array_{ nullptr };
        cudaSurfaceObject_t surf_obj_{ 0 };
        cudaTextureObject_t tex_obj_{ 0 };
    };

    template <class T>
    class CudaSurfaceTexture3D : public CudaSurfaceTexture<T> {
    public:
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, float2> || std::is_same_v<T, float4>
            || std::is_same_v<T, int> || std::is_same_v<T, int2> || std::is_same_v < T, int4>,
            "");

        CudaSurfaceTexture3D() = default;
        ~CudaSurfaceTexture3D()
        {
            Destroy();
        }

        CudaSurfaceTexture3D(const CudaSurfaceTexture3D&) = delete;
        CudaSurfaceTexture3D(CudaSurfaceTexture3D&& other) = delete;
        CudaSurfaceTexture3D& operator=(const CudaSurfaceTexture3D&) = delete;
        CudaSurfaceTexture3D& operator=(CudaSurfaceTexture3D&& other) = delete;

        bool Init(
            int32_t width, int32_t height, int32_t depth,
            aten::TextureFilterMode filter)
        {
            if (surf_obj_ != 0) {
                return false;
            }

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
            cudaExtent extent = make_cudaExtent(width, height, depth);
            auto result = checkCudaErrors(cudaMalloc3DArray(
                &cuda_array_,
                &channelDesc,
                extent,
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

            result = CreateTextureObject(filter);
            if (!result) {
                AT_ASSERT(false);
                Destroy();
                return false;
            }

            return true;
        }
    };
}
