#pragma once

#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten
{
    void generateMipMaps(
        cudaMipmappedArray_t mipmapArray,
        int width, int height,
        int maxLevel);
}
