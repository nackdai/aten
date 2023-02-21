#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

#define BLOCK_SIZE    (16)
#define BLOCK_SIZE2    (BLOCK_SIZE * BLOCK_SIZE)

inline AT_DEVICE_API int32_t getIdx(int32_t ix, int32_t iy, int32_t width)
{
#if 0
    int32_t X = ix / BLOCK_SIZE;
    int32_t Y = iy / BLOCK_SIZE;

    //int32_t base = Y * BLOCK_SIZE2 * (width / BLOCK_SIZE) + X * BLOCK_SIZE2;

    int32_t XB = X * BLOCK_SIZE;
    int32_t YB = Y * BLOCK_SIZE;

    int32_t base = YB * width + XB * BLOCK_SIZE;

    const auto idx = base + (iy - YB) * BLOCK_SIZE + (ix - XB);

    return idx;
#else
    return iy * width + ix;
#endif
}
