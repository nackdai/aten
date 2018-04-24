#pragma once

#include <cuda.h>
#include "cuda/cudautil.h"

namespace idaten {
	void initCuda()
	{
		checkCudaErrors(cuInit(0));
	}
}