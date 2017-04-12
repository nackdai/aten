#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

	void sumArrayOnGPU(float* A, float* B, float* C, const int N);

#ifdef __cplusplus
}
#endif /* __cplusplus */
