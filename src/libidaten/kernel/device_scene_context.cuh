#pragma once

#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"
#include "geometry/geomparam.h"
#include "geometry/vertex.h"
#include "light/light_parameter.h"
#include "material/material.h"
#include "math/mat4.h"
#include "volume/volume_grid.h"

namespace idaten {
    // NOTE:
    // https://stackoverflow.com/questions/43235899/cuda-restrict-tag-usage
    // __restrict__ is hint for the compiler that we use the pointer to refer underlying data.
    // And then, the compiler optimize it.
    //
    // https://forums.developer.nvidia.com/t/restrict-seems-to-be-ignored-for-base-pointers-in-structs-having-base-pointers-with-restrict-as-kernel-arguments-directly-works-as-expected/154020/12
    // __restrict__ doesn't work within a struct as a kernel argument.
    // It works as a kernel argument.
    //
    // Therefore, the readonly data is not applied to the variable in context class and it is specified with __restrict__.
    // __restrict__ can be specified for only pointer.
    // Therefore, the texture data and int variable are applied before calling kernel.

    class context {
    public:
        const aten::ObjectParameter* shapes{ nullptr };

        const aten::MaterialParameter* mtrls{ nullptr };

        int32_t lightnum{ 0 };
        const aten::LightParameter* lights{ nullptr };

        const aten::TriangleParameter* prims{ nullptr };

        const aten::mat4* matrices{ nullptr };

        cudaTextureObject_t vtxPos{ 0 };
        cudaTextureObject_t vtxNml{ 0 };

        cudaTextureObject_t* nodes{ nullptr };

        cudaTextureObject_t* textures{ nullptr };
        int32_t envmapIdx{ -1 };

        idaten::GridHolder grid_holder;

        aten::aabb scene_bounding_box;

        __device__ const float4 GetPosition(uint32_t idx) const noexcept
        {
#ifdef __CUDA_ARCH__
            return tex1Dfetch<float4>(vtxPos, idx);
#else
            return make_float4(0, 0, 0, 0);
#endif
        }

        __device__ const aten::vec4 GetPositionAsVec4(uint32_t idx) const noexcept
        {
            auto v{ GetPosition(idx) };
            return aten::vec4(v.x, v.y, v.z, v.w);
        }

        __device__ const aten::vec4 GetPositionAsVec3(uint32_t idx) const noexcept
        {
            auto v{ GetPosition(idx) };
            return aten::vec3(v.x, v.y, v.z);
        }

        __device__ const float4 GetNormal(uint32_t idx) const noexcept
        {
#ifdef __CUDA_ARCH__
            return tex1Dfetch<float4>(vtxNml, idx);
#else
            return make_float4(0, 0, 0, 0);
#endif
        }

        __device__ const aten::vec4 GetNormalAsVec4(uint32_t idx) const noexcept
        {
            auto v{ GetNormal(idx) };
            return aten::vec4(v.x, v.y, v.z, v.w);
        }

        __device__ const aten::vec4 GetNormalAsVec3(uint32_t idx) const noexcept
        {
            auto v{ GetNormal(idx) };
            return aten::vec3(v.x, v.y, v.z);
        }

        __device__ const aten::ObjectParameter& GetObject(uint32_t idx) const noexcept
        {
            return shapes[idx];
        }

        __device__ const aten::MaterialParameter& GetMaterial(uint32_t idx) const noexcept
        {
            return mtrls[idx];
        }

        __device__ const aten::TriangleParameter& GetTriangle(uint32_t idx) const noexcept
        {
            return prims[idx];
        }

        __device__ const aten::LightParameter& GetLight(uint32_t idx) const noexcept
        {
            return lights[idx];
        }

        __device__ const aten::mat4& GetMatrix(uint32_t idx) const noexcept
        {
            return matrices[idx];
        }

        __device__ int32_t GetLightNum() const noexcept
        {
            return lightnum;
        }

        __device__ cudaTextureObject_t GetTexture(int32_t idx) const noexcept
        {
            return textures[idx];
        }

        __device__ const idaten::GridHolder* GetGrid() const noexcept
        {
            return &grid_holder;
        }

        __device__ const aten::aabb& GetSceneBoundingBox() const
        {
            return scene_bounding_box;
        }
    };

    struct DeviceContextInHost {
        idaten::context ctxt;

        idaten::TypedCudaMemory<aten::ObjectParameter> shapeparam;
        idaten::TypedCudaMemory<aten::MaterialParameter> mtrlparam;
        idaten::TypedCudaMemory<aten::LightParameter> lightparam;
        idaten::TypedCudaMemory<aten::TriangleParameter> primparams;

        idaten::TypedCudaMemory<aten::mat4> mtxparams;

        std::vector<idaten::CudaTextureResource> nodeparam;
        idaten::TypedCudaMemory<cudaTextureObject_t> nodetex;

        std::vector<idaten::CudaTexture> texRsc;
        idaten::TypedCudaMemory<cudaTextureObject_t> tex;

        idaten::TypedCudaMemory<nanovdb::FloatGrid*> grids;

        idaten::CudaTextureResource vtxparamsPos;
        idaten::CudaTextureResource vtxparamsNml;

        void BindToDeviceContext();
    };
}
