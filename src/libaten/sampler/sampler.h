#pragma once

#ifdef __AT_CUDA__

#undef AT_VIRTUAL
#undef AT_VIRTUAL_OVERRIDE_FINAL
#undef AT_INHERIT

#define AT_VIRTUAL(f)                   f
#define AT_VIRTUAL_OVERRIDE_FINAL(f)    f
#define AT_INHERIT(c)

#include "sampler/wanghash.h"
#include "sampler/sobolproxy.h"
#include "sampler/cmj.h"
#include "sampler/bluenoiseSampler.h"

#include "kernel/bluenoiseSampler.cuh"

#define IDATEN_SAMPLER_WANGHASH     (0)
#define IDATEN_SAMPLER_SOBOL        (1)
#define IDATEN_SAMPLER_CMJ          (2)
#define IDATEN_SAMPLER_BLUENOISE    (3)

#define IDATEN_SAMPLER    IDATEN_SAMPLER_CMJ

namespace aten {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    using sampler = Sobol;
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    using sampler = CMJ;
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_WANGHASH
    using sampler = WangHash;
#else
    using sampler = idaten::BlueNoiseSamplerGPU;
#endif
}
#else
#include "samplerinterface.h"
#endif
