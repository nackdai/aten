#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

namespace aten {
	namespace cuda {
		template< typename T >
		void check(T result, char const *const func, const char *const file, int const line)
		{
			if (result)
			{
				fprintf(
					stderr,
					"CUDA error at %s:%d code=%d(%s) \"%s\" \n",
					file,
					line,
					static_cast<unsigned int>(result),
					//_cudaGetErrorEnum(result),
					cudaGetErrorString(result),
					func);

				cudaDeviceReset();

				// Make sure we call CUDA Device Reset before exiting.
				exit(EXIT_FAILURE);
			}
		}
	}
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error.
#define checkCudaErrors(val)	aten::cuda::check((val), #val, __FILE__, __LINE__)
