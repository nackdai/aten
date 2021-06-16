#pragma once

#include "aten4idaten.h"

namespace idaten {
    struct Context {
        int geomnum{ 0 };
        const aten::GeomParameter* shapes{ nullptr };

        const aten::MaterialParameter* mtrls{ nullptr };

        int lightnum{ 0 };
        const aten::LightParameter* lights{ nullptr };

        cudaTextureObject_t* nodes{ nullptr };

        const aten::PrimitiveParamter* prims{ nullptr };

        cudaTextureObject_t vtxPos{ 0 };
        cudaTextureObject_t vtxNml{ 0 };

        const aten::mat4* matrices{ nullptr };

        cudaTextureObject_t* textures{ nullptr };
        int envmapIdx{ -1 };
    };
}
