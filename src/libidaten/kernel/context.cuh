#pragma once

#include "aten4idaten.h"

namespace idaten {
    struct Context {
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
    };
}
