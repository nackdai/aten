#pragma once

#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "cuda/cudaTextureResource.h"
#include "geometry/vertex.h"
#include "material/material.h"
#include "light/light_parameter.h"
#include "geometry/geomparam.h"
#include "math/mat4.h"

namespace idaten {
    // NOTE:
    // https://forums.developer.nvidia.com/t/restrict-seems-to-be-ignored-for-base-pointers-in-structs-having-base-pointers-with-restrict-as-kernel-arguments-directly-works-as-expected/154020/12
    // __restrict__ hint doesn't work within a struct as a kernel argument.
    // It works as a kernel argument.

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

        idaten::CudaTextureResource vtxparamsPos;
        idaten::CudaTextureResource vtxparamsNml;

        void BindToDeviceContext()
        {
            if (!ctxt.shapes) {
                ctxt.lightnum = static_cast<int32_t>(lightparam.num());

                std::vector<cudaTextureObject_t> tmp_node;
                for (auto& node : nodeparam) {
                    auto nodeTex = node.bind();
                    tmp_node.push_back(nodeTex);
                }
                nodetex.writeFromHostToDeviceByNum(tmp_node.data(), tmp_node.size());

                ctxt.nodes = nodetex.data();

                if (!texRsc.empty())
                {
                    std::vector<cudaTextureObject_t> tmp_tex;
                    for (auto& rsc : texRsc) {
                        auto cudaTex = rsc.bind();
                        tmp_tex.push_back(cudaTex);
                    }
                    tex.writeFromHostToDeviceByNum(tmp_tex.data(), tmp_tex.size());
                }
                ctxt.textures = tex.data();
            }

            ctxt.vtxPos = vtxparamsPos.bind();
            ctxt.vtxNml = vtxparamsNml.bind();

            for (auto& node : nodeparam) {
                std::ignore = node.bind();
            }
        }
    };
}
