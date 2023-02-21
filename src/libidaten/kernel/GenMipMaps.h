#pragma once

#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten
{
    void generateMipMaps(
        cudaMipmappedArray_t mipmapArray,
        int32_t width, int32_t height,
        int32_t maxLevel);
}
