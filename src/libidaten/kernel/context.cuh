#pragma once

#include "aten4idaten.h"

struct Context {
    int geomnum;
    const aten::GeomParameter* shapes;

    const aten::MaterialParameter* mtrls;

    int lightnum;
    const aten::LightParameter* lights;

    cudaTextureObject_t* nodes;

    const aten::PrimitiveParamter* prims;

    cudaTextureObject_t vtxPos;
    cudaTextureObject_t vtxNml;

    const aten::mat4* matrices;

    cudaTextureObject_t* textures;
    int envmapIdx;
};
