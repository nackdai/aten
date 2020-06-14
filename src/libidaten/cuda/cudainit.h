#pragma once

#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"

namespace idaten {
    inline void initCuda()
    {
        checkCudaErrors(cuInit(0));
    }
}
