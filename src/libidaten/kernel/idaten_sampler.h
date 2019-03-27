#pragma once

#include "sampler/sampler.h"

#include "kernel/bluenoiseSampler.cuh"

namespace aten {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_BLUENOISE
    using sampler = idaten::BlueNoiseSampler;
#endif
}