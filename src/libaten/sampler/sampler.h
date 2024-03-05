#pragma once

#ifdef __AT_CUDA__

#undef AT_VIRTUAL
#undef AT_VIRTUAL_OVERRIDE_FINAL
#undef AT_VIRTUAL_OVERRIDE
#undef AT_INHERIT

#define AT_VIRTUAL(f)                   f
#define AT_VIRTUAL_OVERRIDE_FINAL(f)    f
#define AT_VIRTUAL_OVERRIDE(f)          f
#define AT_INHERIT(c)
#endif

#include "sampler/wanghash.h"
#include "sampler/cmj.h"

#define IDATEN_SAMPLER_WANGHASH     (0)
#define IDATEN_SAMPLER_CMJ          (1)

#define IDATEN_SAMPLER    IDATEN_SAMPLER_CMJ

namespace aten {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    using sampler = CMJ;
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_WANGHASH
    using sampler = WangHash;
#endif
}
