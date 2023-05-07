#pragma once

#include "aten4idaten.h"

namespace idaten {
    struct Context {
        const aten::ObjectParameter* shapes{ nullptr };

        const aten::MaterialParameter* mtrls{ nullptr };

        int32_t lightnum{ 0 };
        const aten::LightParameter* lights{ nullptr };

        cudaTextureObject_t* nodes{ nullptr };

        const aten::TriangleParameter* prims{ nullptr };

        cudaTextureObject_t vtxPos{ 0 };
        cudaTextureObject_t vtxNml{ 0 };

        const aten::mat4* matrices{ nullptr };

        cudaTextureObject_t* textures{ nullptr };
        int32_t envmapIdx{ -1 };
    };
}
