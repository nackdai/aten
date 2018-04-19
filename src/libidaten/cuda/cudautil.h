#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"

namespace idaten {
	namespace cuda {
		template< typename T >
		void check(T result, char const *const func, const char *const file, int const line)
		{
			if (result)
			{
				static char buf[2048];

#if 0
				fprintf(
					stderr,
					"CUDA error at %s:%d code=%d(%s) \"%s\" \n",
					file,
					line,
					static_cast<unsigned int>(result),
					//_cudaGetErrorEnum(result),
					cudaGetErrorString(result),
					func);
#else
				snprintf(
					buf, 2048,
					"CUDA error at %s:%d code=%d(%s) \"%s\" \n",
					file,
					line,
					static_cast<unsigned int>(result),
					//_cudaGetErrorEnum(result),
					cudaGetErrorString(result),
					func);

#if 0
				fprintf(stderr, "%s", buf);
				::OutputDebugString(buf);
#else
				AT_PRINTF("%s", buf);
#endif

				AT_ASSERT(false);
#endif

				cudaDeviceReset();

				// Make sure we call CUDA Device Reset before exiting.
				exit(EXIT_FAILURE);
			}
		}
	}


}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error.
#define checkCudaErrors(val)	idaten::cuda::check((val), #val, __FILE__, __LINE__)

#ifdef __AT_DEBUG__
#define checkCudaKernel(kernel)	{\
	auto err = cudaGetLastError();\
	if (err != cudaSuccess) {\
		AT_PRINTF("Cuda Kernel Err(%s) [%s]\n", (#kernel), cudaGetErrorString(err));\
	}\
	err = cudaDeviceSynchronize();\
	if (cudaSuccess != err) {\
		AT_PRINTF("Cuda Kernel Err with Sync(%s) [%s]\n", (#kernel), cudaGetErrorString(err));\
	}\
}
#else
#define checkCudaKernel(kernel)
#endif
