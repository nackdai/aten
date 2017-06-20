#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#include "aten4idaten.h"

#define BLOCK_SIZE	(16)
#define BLOCK_SIZE2	(BLOCK_SIZE * BLOCK_SIZE)

inline AT_DEVICE_API int getIdx(int ix, int iy, int width)
{
	int X = ix / BLOCK_SIZE;
	int Y = iy / BLOCK_SIZE;

	//int base = Y * BLOCK_SIZE2 * (width / BLOCK_SIZE) + X * BLOCK_SIZE2;

	int XB = X * BLOCK_SIZE;
	int YB = Y * BLOCK_SIZE;

	int base = YB * width + XB * BLOCK_SIZE;

	const auto idx = base + (iy - YB) * BLOCK_SIZE + (ix - XB);

	return idx;
}

