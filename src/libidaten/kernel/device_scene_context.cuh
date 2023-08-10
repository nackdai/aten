#pragma once

#include "geometry/vertex.h"
#include "material/material.h"
#include "light/light_parameter.h"
#include "geometry/geomparam.h"
#include "math/mat4.h"

namespace idaten {
    struct context {
        const aten::ObjectParameter* __restrict__ shapes{ nullptr };

        const aten::MaterialParameter* __restrict__ mtrls{ nullptr };

        int32_t lightnum{ 0 };
        const aten::LightParameter* __restrict__ lights{ nullptr };

        const aten::TriangleParameter* __restrict__ prims{ nullptr };

        const aten::mat4* __restrict__ matrices{ nullptr };

        cudaTextureObject_t vtxPos{ 0 };
        cudaTextureObject_t vtxNml{ 0 };

        cudaTextureObject_t* nodes{ nullptr };

        cudaTextureObject_t* textures{ nullptr };
        int32_t envmapIdx{ -1 };

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
            auto v = GetPosition(idx);
            return aten::vec4(v.x, v.y, v.z, v.w);
        }

        __device__ const aten::vec4 GetPositionAsVec3(uint32_t idx) const noexcept
        {
            auto v = GetPosition(idx);
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
            auto v = GetNormal(idx);
            return aten::vec4(v.x, v.y, v.z, v.w);
        }

        __device__ const aten::vec4 GetNormalAsVec3(uint32_t idx) const noexcept
        {
            auto v = GetNormal(idx);
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

        __device__ int32_t get_light_num() const noexcept
        {
            return lightnum;
        }
    };
}
