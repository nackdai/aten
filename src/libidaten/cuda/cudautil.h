#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <utility>

namespace idaten {
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

			fprintf(stderr, "%s", buf);
			::OutputDebugString(buf);
#endif

			cudaDeviceReset();

			// Make sure we call CUDA Device Reset before exiting.
			exit(EXIT_FAILURE);
		}
	}
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error.
#define checkCudaErrors(val)	idaten::check((val), #val, __FILE__, __LINE__)
